import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = qk_scale or (dim // num_heads) ** -0.5
        self.with_qkv  = with_qkv
        if with_qkv:
            self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj      = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp   = Mlp(in_features=dim, hidden_features=int(mlp_dim),
                         act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_features=100, embed_dim=768, depth=12, num_heads=12,
                 mlp_dim=2048, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = num_features
        self.pos_embed    = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        self.pos_drop     = nn.Dropout(p=drop_rate)
        self.blocks       = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if x.size(1) != self.pos_embed.size(1):
            pe = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1), mode='nearest'
            ).transpose(1, 2)
            x = x + pe
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def __init__(self, num_features=100, embed_dim=768, depth=12, num_heads=12,
                 mlp_dim=2048, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = num_features
        self.pos_embed    = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        self.pos_drop     = nn.Dropout(p=drop_rate)
        self.blocks       = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        if x.size(1) != self.pos_embed.size(1):
            pe = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1), mode='nearest'
            ).transpose(1, 2)
            x = x + pe
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)