import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vit import VisionTransformer
from model import t5


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(drop)
        self.scale = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.drop(x)
        return x * self.scale


class DenseVideoCaption(nn.Module):
    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 dec_drop=0.1,
                 num_bins=100):
        super().__init__()

        # FIX 1: lưu num_bins để dùng trong forward()
        self.num_bins = num_bins

        self.visual_encoder = VisionTransformer(
            num_features=num_features,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            mlp_dim=mlp_dim,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=vis_drop,
            attn_drop_rate=vis_drop,
            norm_layer=nn.LayerNorm
        )

        self.tokenizer = t5.load_tokenizer(t5_path, num_bins=num_bins)
        self.t5_model = t5.load_model(self.tokenizer, t5_path)

        self.projection = Projection(
            in_dim=embed_dim,
            out_dim=self.t5_model.config.d_model,
            drop=dec_drop
        )

        # Cache time token ids — tránh tính lại mỗi forward pass
        self._time_token_ids = None

    def _get_time_token_ids(self, device):
        if self._time_token_ids is None:
            ids = [
                self.tokenizer.convert_tokens_to_ids(f"<times={i}>")
                for i in range(self.num_bins)
            ]
            self._time_token_ids = torch.tensor(ids)
        return self._time_token_ids.to(device)

    def forward(self, video_features, label_input_ids, label_attention_mask):
        """Forward pass with pre-tokenized inputs.
        
        Args:
            video_features: Dict with 'video_features' and 'attention_mask', or just tensor
            label_input_ids: Pre-tokenized input IDs (B, T)
            label_attention_mask: Attention mask for labels (B, T)
        """
        if isinstance(video_features, dict):
            attention_mask = video_features['attention_mask']
            video_features = video_features['video_features']
        else:
            attention_mask = None

        visual_embeds = self.visual_encoder(video_features)
        visual_embeds = self.projection(visual_embeds)

        if attention_mask is None:
            attention_mask = torch.ones(
                visual_embeds.size()[:-1],
                dtype=torch.long,
                device=visual_embeds.device
            )

        # FIX: Use pre-tokenized input_ids directly (no redundant tokenization)
        input_ids = label_input_ids.to(visual_embeds.device)
        # Ensure padding tokens are masked as -100 for loss calculation
        input_ids = input_ids.clone()
        input_ids[input_ids == self.tokenizer.pad_token_id] = -100

        # Per-token loss để weight riêng time tokens
        logits = self.t5_model(
            inputs_embeds=visual_embeds,
            attention_mask=attention_mask,
            labels=input_ids
        ).logits  # (B, T, vocab)

        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction='none',
            ignore_index=-100
        ).view(input_ids.shape)  # (B, T)

        # Time token weight 2.0: ép model học timestamp tốt hơn
        time_token_ids = self._get_time_token_ids(input_ids.device)
        is_time_token = torch.isin(input_ids, time_token_ids)
        weight = torch.where(is_time_token,
                             torch.tensor(2.0, device=input_ids.device),
                             torch.tensor(1.0, device=input_ids.device))

        valid = (input_ids != -100).float()
        loss = (per_token_loss * weight * valid).sum() / valid.sum()

        return {"loss": loss}

    @torch.no_grad()
    def generate(self, video_features, max_length=200, num_beams=5):
        self.eval()

        # FIX 2: handle dict input (batch có padding)
        if isinstance(video_features, dict):
            attention_mask = video_features['attention_mask']
            video_features = video_features['video_features']
        else:
            attention_mask = None

        visual_embeds = self.visual_encoder(video_features)
        visual_embeds = self.projection(visual_embeds)

        if attention_mask is None:
            attention_mask = torch.ones(
                visual_embeds.size()[:-1],
                dtype=torch.long,
                device=visual_embeds.device
            )

        outputs = self.t5_model.generate(
            inputs_embeds=visual_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            length_penalty=0.8,
        )

        # FIX 3: skip_special_tokens=False để giữ <times=N> trong output
        captions = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=False
        )

        # Chỉ xóa padding token, giữ lại time tokens
        pad_token = self.tokenizer.pad_token
        captions = [c.replace(pad_token, '').strip() for c in captions]

        return captions