import math

import torch
import torch.nn.functional as F


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape
    for i in range(b):
        bbox_i = bbox[i].cpu().tolist()
        bbox_mask[i, int(bbox_i[1]):int(bbox_i[1] + bbox_i[3] - 1), int(bbox_i[0]):int(bbox_i[0] + bbox_i[2] - 1)] = 1
    return bbox_mask


def generate_mask_cond(cfg, bs, device, gt_bbox):
    template_size = cfg.DATA.TEMPLATE.SIZE
    stride = cfg.MODEL.BACKBONE.STRIDE
    template_feat_size = template_size // stride

    if cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'ALL':
        box_mask_z = None
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT':
        if template_feat_size == 8:
            index = slice(3, 4)
        elif template_feat_size == 12:
            index = slice(5, 6)
        elif template_feat_size == 7:
            index = slice(3, 4)
        elif template_feat_size == 14:
            index = slice(6, 7)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_REC':
        if template_feat_size == 8:
            index = slice(3, 5)
        elif template_feat_size == 12:
            index = slice(5, 7)
        elif template_feat_size == 7:
            index = slice(3, 4)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'GT_BOX':
        box_mask_z = torch.zeros([bs, template_size, template_size], device=device)
        box_mask_z = generate_bbox_mask(box_mask_z, gt_bbox * template_size).unsqueeze(1).to(
            torch.float)  
        box_mask_z = F.interpolate(box_mask_z, scale_factor=1. / cfg.MODEL.BACKBONE.STRIDE, mode='bilinear',
                                   align_corners=False)
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    else:
        raise NotImplementedError

    return box_mask_z


def adjust_keep_rate(epoch, warmup_epochs, total_epochs):
    depth = 12
    if epoch < warmup_epochs:
        return [1]
    else:
        keep_rate =[x for x in torch.linspace(0.7, 1, depth//4)][::-1]

    return keep_rate
