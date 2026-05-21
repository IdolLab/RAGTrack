from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd

def candidate_elimination(tokens: torch.Tensor, attn: torch.Tensor, template_mask: torch.Tensor, len_text: int, lens_t: int, lens_s: int, keep_ratio: float):
    
    lens_q = attn.shape[-1] - len_text - lens_t - lens_s
    bs, hn, _, _ = attn.shape
    lens_keep = math.ceil(keep_ratio * lens_s)

    tokens_q = tokens[:, :lens_q, :]
    tokens_text = tokens[:, lens_q:lens_q + len_text, :]
    tokens_t = tokens[:,lens_q + len_text:lens_t + lens_q + len_text,:]
    tokens_s = tokens[:,lens_q + len_text + lens_t:,:]

    attn_t = attn[:, :, lens_q + len_text:lens_t + lens_q + len_text, lens_q + len_text + lens_t:]
    if template_mask is not None:
        template_mask = template_mask.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[template_mask]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)

    attn_q = attn[:, :, :lens_q, lens_q + len_text + lens_t:]
    attn_q = attn_q.mean(dim=2).mean(dim=1)

    attn_text = attn[:, :, lens_q:lens_q + len_text, lens_q + len_text + lens_t:]
    attn_text = attn_text.mean(dim=2).mean(dim=1)

    attn_s = attn[:, :, lens_q + len_text + lens_t:, lens_q + len_text + lens_t:]
    attn_s = attn_s.mean(dim=2).mean(dim=1)

    attn_vision = attn_q + attn_t + attn_text + attn_s

    token_mask = torch.ones_like(tokens_s).to('cuda')
    zeros_mask = torch.zeros_like(tokens_s).to('cuda')

    sorted_attn, indices = torch.sort(attn_vision, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    global_index = torch.arange(lens_s).expand(bs, -1).to('cuda')
    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    
    token_mask.scatter_(dim=1, index=non_topk_idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)), src=zeros_mask)
    tokens_s_new = tokens_s * token_mask

    tokens_new = torch.cat([tokens_q, tokens_text, tokens_t, tokens_s_new],dim=1)

    return tokens_new, removed_index


class AdaptiveChannelExchange(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable projections for Query (RGB) and Key (TIR)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x_rgb, x_tir):
        B, N, C = x_rgb.shape

        # Compute channel-wise correlation matrix (B, N, C)
        q = self.W_Q(x_rgb)  # (B, N, C)
        k = self.W_K(x_tir)  # (B, N, C)

        x_rgb_t = q.transpose(1, 2)  # (B, C, N)
        x_tir_t = k.transpose(1, 2)  # (B, C, N)

        att_matrix = x_rgb_t @ x_tir_t.transpose(1, 2)
        att_matrix = att_matrix.mean(dim=1)

        sorted_attn, indices = torch.sort(att_matrix, dim=1, descending=True)
        topk_idx = indices[:, :C // 2]
        token_mask = torch.ones_like(x_rgb_t)  # (B, C, N)
        switch_weights = torch.zeros_like(x_rgb_t)  # (B, C, N)
        topk_idx_expanded = topk_idx.unsqueeze(2).expand(-1, -1, N)  # (B, 1, N)
        switch_weights.scatter_(dim=1, index=topk_idx_expanded, src=token_mask)  # (B, C, N)
        x_rgb_swapped = (1.0 - switch_weights) * x_rgb_t + switch_weights * x_tir_t
        x_tir_swapped = switch_weights * x_rgb_t + (1.0 - switch_weights) * x_tir_t
        x_rgb_swapped = x_rgb_swapped.transpose(1, 2)
        x_tir_swapped = x_tir_swapped.transpose(1, 2)

        return x_rgb_swapped, x_tir_swapped

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Up_Down(nn.Module):
    def __init__(self, dim):
        super().__init__()
 
        self.adapter_down = nn.Linear(dim, 8)
        self.adapter_up = nn.Linear(8, dim)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x_r, x_x):
        x = torch.cat([x_r, x_x], dim=-2)
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        x_r_adap = x_up[:, :N // 2, :]
        x_x_adap = x_up[:, N // 2:, :]

        return x_r_adap, x_x_adap