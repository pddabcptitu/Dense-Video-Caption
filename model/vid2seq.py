import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from models.vit import VisionTransformer

# ═══════════════════════════════════════════════════════════════
# Vid2Seq model
# ═══════════════════════════════════════════════════════════════
class Vid2Seq(nn.Module):
    def __init__(self, t5_path, num_features=100, embed_dim=768, depth=12,
                 heads=12, mlp_dim=2048, vis_drop=0., tokenizer=None, num_bins=100,
                 contrastive_dim=256, contrastive_weight=0.1):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_path)
        self.t5.resize_token_embeddings(len(tokenizer) - num_bins)
        self.t5.resize_token_embeddings(len(tokenizer))

        # Freeze T5 encoder — only decoder + visual encoder learn
        for param in self.t5.encoder.parameters():
            param.requires_grad = False

        self.visual_encoder = VisionTransformer(
            num_features=num_features, embed_dim=embed_dim, depth=depth,
            num_heads=heads, mlp_dim=mlp_dim, qkv_bias=True, qk_scale=None,
            drop_rate=vis_drop, attn_drop_rate=vis_drop, norm_layer=nn.LayerNorm,
        )
        d_model = self.t5.config.d_model
        self.proj_v2t = (
            nn.Linear(embed_dim, d_model)
            if d_model != embed_dim else None
        )
        self.tokenizer = tokenizer

        # ── Contrastive heads ──────────────────────────────────────────────
        # Project video mean-pooled features → contrastive space
        self.video_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, contrastive_dim),
        )
        # Project T5 decoder hidden-state mean → contrastive space
        # T5 decoder output dim == d_model
        self.text_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, contrastive_dim),
        )
        # Learnable temperature for InfoNCE
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)  # ln(14) ≈ 2.659
        self.contrastive_weight = contrastive_weight

    # ─────────────────────────────────────────────
    def _encode_video(self, video):
        video    = self.visual_encoder(video)           # (B, T, embed_dim)
        if self.proj_v2t is not None:
            video = self.proj_v2t(video)               # (B, T, d_model)
        atts_vis = torch.ones(video.shape[:2], dtype=torch.long, device=video.device)
        return BaseModelOutput(last_hidden_state=video), atts_vis

    # ─────────────────────────────────────────────
    @staticmethod
    def _info_nce(video_emb, text_emb, logit_scale):
        """
        Symmetric InfoNCE (CLIP-style) contrastive loss.

        Args:
            video_emb : (B, D) L2-normalised video embeddings
            text_emb  : (B, D) L2-normalised text embeddings
            logit_scale: scalar temperature parameter

        Returns:
            scalar loss
        """
        # Clamp temperature to avoid numerical explosion
        scale  = logit_scale.exp().clamp(max=100.0)
        logits = scale * video_emb @ text_emb.t()           # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_v = F.cross_entropy(logits,   labels)          # video → text
        loss_t = F.cross_entropy(logits.t(), labels)        # text  → video
        return (loss_v + loss_t) / 2.0

    # ─────────────────────────────────────────────
    def forward(self, video, output_tokenized):
        encoded, atts_vis = self._encode_video(video)

        targets = output_tokenized["input_ids"].masked_fill(
            output_tokenized["input_ids"] == self.tokenizer.pad_token_id, -100
        )

        # T5 forward — encoder_outputs bypasses frozen T5 encoder
        out = self.t5(
            encoder_outputs=encoded,
            attention_mask=atts_vis,
            decoder_attention_mask=output_tokenized["attention_mask"],
            labels=targets,
            return_dict=True,
            output_hidden_states=True,          # need decoder hidden states
        )
        caption_loss = out.loss

        # ── Contrastive loss ────────────────────────────────────────────
        # video_emb: mean-pool over temporal dim of visual encoder output
        video_emb = encoded.last_hidden_state.mean(dim=1)   # (B, d_model)
        video_emb = self.video_proj(video_emb)              # (B, contrastive_dim)
        video_emb = F.normalize(video_emb, dim=-1)

        # text_emb: mean-pool over sequence dim of last decoder hidden state
        # decoder_hidden_states[-1] shape: (B, seq_len, d_model)
        dec_hidden = out.decoder_hidden_states[-1]          # (B, seq_len, d_model)
        # Mask out padding tokens before mean-pooling
        pad_mask   = output_tokenized["attention_mask"].unsqueeze(-1).float()  # (B, seq, 1)
        text_emb   = (dec_hidden * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1e-6)
        text_emb   = self.text_proj(text_emb)               # (B, contrastive_dim)
        text_emb   = F.normalize(text_emb, dim=-1)

        contrastive_loss = self._info_nce(video_emb, text_emb, self.logit_scale)

        total_loss = caption_loss + self.contrastive_weight * contrastive_loss

        return {
            "loss":             total_loss,
            "caption_loss":     caption_loss,
            "contrastive_loss": contrastive_loss,
        }

    @torch.no_grad()
    def generate(self, video, num_beams=4, max_length=256, min_length=1,
                 use_nucleus_sampling=False, top_p=0.9, repetition_penalty=1.0,
                 length_penalty=1.0, num_captions=1, temperature=1.0):
        encoded, atts_vis = self._encode_video(video)
        ids = self.t5.generate(
            encoder_outputs=encoded,
            attention_mask=atts_vis,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=False)
