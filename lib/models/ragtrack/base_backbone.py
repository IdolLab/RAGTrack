from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import Parameter

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ragtrack.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    temporal_token: Parameter

    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = True
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER

        # patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = self.pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size // self.patch_size, self.img_size // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        #temporal token
        self.temporal_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.add_cls_token:
            temporal_pos_embed = self.pos_embed[:, 0:1, :]
            self.temporal_pos_embed = nn.Parameter(temporal_pos_embed)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, l, temporal_query=None):
        B = x.shape[0]
        descript_id = self.tokenizer(l, add_special_tokens=False, truncation=True, pad_to_max_length=True,max_length=16)['input_ids']
        descript_id_tensor = torch.tensor(descript_id)
        if self.add_cls_token:
            temporal_init = self.temporal_token.expand(B,1,-1)
            temporal_init = temporal_init + self.temporal_pos_embed

        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)
        x = self.patch_embed(x)
        l = self.descript_embedding(descript_id_tensor.to(x.device))

        if not self.convmlp and self.stop_grad_conv1:  # self.convmlp==True
            x = x.detach() * 0.9 + x * 0.1
            assert self.convmlp == True, '想像失败'


        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)
            z = blk(z)
            # l = blk(l)

        # x = x[..., 0, 0, :]
        # z = z[..., 0, 0, :]

        x = x.flatten(2).transpose(1, 2)
        z = z.flatten(2).transpose(1, 2)

        z += self.pos_embed_z
        x += self.pos_embed_x
        l += self.description_patch_pos_embed(l)

        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = combine_tokens(l, x,  mode=self.cat_mode)

        if self.add_cls_token:
            if temporal_query is None:
                x = torch.cat([temporal_init, x], dim=1)
            else:
                x = torch.cat([temporal_query, x], dim=1)


        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        assert rel_pos_bias == None, 'rel_pos_bias not None'
        assert self.grad_ckpt == False, 'grad_ckpt != Fasle'
        for blk in self.blocks[-self.num_main_blocks:]:
            x = blk(x)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        x = self.norm(x)

        return x, aux_dict


    def forward(self, z, x, track_query_before, text_label, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x, track_query_before, text_label)

        return x, aux_dict
