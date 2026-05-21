import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.utils.box_ops import box_xyxy_to_cxcywh
from timm.models.layers import Mlp
from lib.models.layers.attn import Attention_qkv
from functools import partial
import torch.nn.functional as F
from lib.models.ragtrack.itpn import fast_itpn_base_3324_patch16_224

class ATTFu(nn.Module):
    def __init__(self, channels, ratio=1.0):
        super(ATTFu, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(4 * channels, int(ratio * channels), bias=False),
            nn.ReLU(),
            nn.Linear(int(ratio * channels), 4 * channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, l_pro, l_text, l_tem, l_tem_d):
        l_tem = torch.mean(l_tem, dim=1, keepdim=True)
        l_tem_d = torch.mean(l_tem_d, dim=1, keepdim=True)
        l_fu = torch.cat([l_pro, l_text, l_tem, l_tem_d], dim=2)
        att = self.fc(l_fu)

        return att[:, :, :self.channels]

class RAGTrack(nn.Module):

    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER"):

        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.box_head = box_head
        self.offset_range_factor = cfg.MODEL.TSG.LAYER
        self.track_query_len = cfg.MODEL.TSG.TRACK_QUERY
        self.track_beforequery_len = cfg.MODEL.TSG.TRACK_QUERY_OLD
        self.template_number = cfg.DATA.TEMPLATE.NUMBER
        self.decode_fuse_search = conv(hidden_dim * 2, hidden_dim)

        self.prompt = ATTFu(hidden_dim)

        self.cos_sim_threshold = 1.0
        self.topk = 2
        self.max_pool_size = 4

        self.text_pool_r = []
        self.text_pool_x = []
        self.text_strengthen_r = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.text_strengthen_x = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)

        self.CSS_strengthen_r = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.CSS_process_r = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4.), act_layer=nn.GELU,
                                 drop=0.)
        self.CSS_strengthen_x = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.CSS_process_x = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4.), act_layer=nn.GELU,
                                 drop=0.)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text_label = None,
                ce_template_mask = None,
                return_last_attn = False,
                track_query_before = None,
                ):

        out_dict = []
        if track_query_before is None:
            self.text_pool_r = []
            self.text_pool_x = []
        for i in range(len(search)):
            x, aux_dict, len_zx = self.backbone(z=template, x=search[i], track_query_before=track_query_before,
                                                text_label=text_label[i],
                                                ce_template_mask=ce_template_mask,
                                                return_last_attn=return_last_attn, )
            num_template_token = len_zx[0]
            num_search_token = len_zx[1]
            B, N, C = x.size()
            temp_r = x[:, :N // 2, :]
            temp_x = x[:, N // 2:, :]
            text_r = temp_r[:, self.track_query_len:self.track_query_len + 1, :]
            text_x = temp_x[:, self.track_query_len:self.track_query_len + 1, :]
            temp_r_str = self.prompt(temp_r[:, :self.track_query_len, :],
                                     text_r,
                                     temp_r[:, self.track_query_len + 1:num_template_token // 2 + self.track_query_len + 1, :],
                                     temp_r[:, num_template_token // 2 + self.track_query_len + 1:num_template_token +self.track_query_len + 1,:])
            temp_x_str = self.prompt(temp_x[:, :self.track_query_len, :],
                                     text_x,
                                     temp_x[:, self.track_query_len + 1:num_template_token // 2 + self.track_query_len + 1, :],
                                     temp_x[:, num_template_token // 2 + self.track_query_len + 1:num_template_token + self.track_query_len + 1,:])
            temp_r_query = temp_r_str.clone().detach()
            temp_x_query = temp_x_str.clone().detach()
            track_query_before = [temp_r_query, temp_x_query]

            feat_last_r = temp_r[:, -num_search_token:, :]
            if len(self.text_pool_r) == 0:
                self.text_pool_r.append(text_r)
            else:
                text_pool_r = torch.cat(self.text_pool_r, dim=1)
                sim_r = F.cosine_similarity(text_r, text_pool_r, dim=-1).mean()
                if sim_r.max() < self.cos_sim_threshold:
                    self.text_pool_r.append(text_r)
            if len(self.text_pool_r) > self.max_pool_size:
                del self.text_pool_r[1]
            if len(self.text_pool_r) > 0:
                text_pool_r = torch.cat(self.text_pool_r, dim=1)
                sim_s_r = F.cosine_similarity(text_r, text_pool_r, dim=-1)
                topk_indices_r = torch.topk(sim_s_r, k=min(self.topk, len(self.text_pool_r)),
                                        largest=True).indices
                retrieved_texts_r = torch.gather(text_pool_r, 1, topk_indices_r.unsqueeze(-1).expand(-1, -1, C))
                feat_last_r = feat_last_r + self.text_strengthen_r(feat_last_r, retrieved_texts_r, retrieved_texts_r)
            temp_attn_r = temp_r_str + self.CSS_strengthen_r(temp_r_str, feat_last_r, feat_last_r)
            temp_attn_r = temp_attn_r + self.CSS_process_r(temp_attn_r)
            att_r = torch.matmul(feat_last_r, temp_attn_r.transpose(1, 2))
            feat_last_r = att_r * feat_last_r

            feat_last_x = temp_x[:, -num_search_token:, :]
            if len(self.text_pool_x) == 0:
                self.text_pool_x.append(text_x)
            else:
                text_pool_x = torch.cat(self.text_pool_x, dim=1)
                sim_x = F.cosine_similarity(text_x, text_pool_x, dim=-1).mean()
                if sim_x.max() < self.cos_sim_threshold:
                    self.text_pool_x.append(text_x)
            if len(self.text_pool_x) > self.max_pool_size:
                del self.text_pool_x[1]
            if len(self.text_pool_x) > 0:
                text_pool_x = torch.cat(self.text_pool_x, dim=1)
                sim_s_x = F.cosine_similarity(text_x, text_pool_x, dim=-1)
                topk_indices_x = torch.topk(sim_s_x, k=min(self.topk, len(self.text_pool_x)),
                                        largest=True).indices
                retrieved_texts_x = torch.gather(text_pool_x, 1, topk_indices_x.unsqueeze(-1).expand(-1, -1, C))
                feat_last_x = feat_last_x + self.text_strengthen_r(feat_last_x, retrieved_texts_x, retrieved_texts_x)
            temp_attn_x = temp_x_str + self.CSS_strengthen_x(temp_x_str, feat_last_x, feat_last_x)
            temp_attn_x = temp_attn_x + self.CSS_process_x(temp_attn_x)
            att_x = torch.matmul(feat_last_x, temp_attn_x.transpose(1, 2))
            feat_last_x = att_x * feat_last_x

            feat_last = torch.cat([feat_last_r, feat_last_x], dim=-1)

            out = self.forward_head(feat_last, None)

            out.update(aux_dict)
            out['track_query_before'] = track_query_before
            out['backbone_feat'] = feat_last
            out_dict.append(out)

        return out_dict

    def forward_head(self, cat_feature ,gt_score_map=None):

        opt = (cat_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        # HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.decode_fuse_search(opt_feat)
        # bs, C, _, _ = opt_feat.size()
        # Nq = 1

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ragtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('DUTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'itpn_base':
        backbone = fast_itpn_base_3324_patch16_224(pretrained, cfg=cfg)
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = RAGTrack(
        backbone,
        box_head,
        cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'DUTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys', missing_keys, unexpected_keys)
    return model
