import torch
import torch.nn as nn
from model.vit import VisionTransformer
from model import t5
from transformers.modeling_outputs import BaseModelOutput


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.drop(x)
        return x


class DenseVideoCation(nn.Module):
    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 tokenizer=None,
                 enc_drop=0.,
                 dec_drop=0.1,
                 use_speech=True,
                 use_video=True,
                 num_bins=0,
                 label_smoothing=0.1):
        super().__init__()

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
            out_dim=self.t5_model.config.d_model 
        )

        self.use_video = use_video


    def forward(self, video_features, labels: str):
        if isinstance(video_features, dict):
            attention_mask = video_features['attention_mask']
            video_features = video_features['video_features']
            visual_embeds = self.visual_encoder(video_features)

        else:
            visual_embeds = self.visual_encoder(video_features)
            attention_mask = torch.ones(
                visual_embeds.size()[:-1],
                dtype=torch.long,
                device=visual_embeds.device
            )
        visual_embeds = self.projection(visual_embeds)

        tokenized = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids.to(visual_embeds.device)

        input_ids[input_ids == self.tokenizer.pad_token_id] = -100
        
        encoder_outputs = BaseModelOutput(
            last_hidden_state=visual_embeds
        )
        
        outputs = self.t5_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(self, video_features, max_length=512, num_beams=3):
        self.eval()

        # 1. Encode video
        visual_embeds = self.visual_encoder(video_features)

        # 2. Projection sang T5 dim
        visual_embeds = self.projection(visual_embeds)

        # 3. Attention mask
        attention_mask = torch.ones(
            visual_embeds.size()[:-1],
            dtype=torch.long,
            device=visual_embeds.device
        )

        encoder_outputs = BaseModelOutput(
            last_hidden_state=visual_embeds
        )
        
        outputs = self.t5_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

        # 5. Decode
        captions = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        return captions