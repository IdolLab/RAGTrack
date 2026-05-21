import math
import numpy as np
from lib.models.ragtrack import build_ragtrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from Qwen_VL.qwen_generate import TextGenerator


class RAGTrack(BaseTracker):
    _sequence_id_counter = 0
    def __init__(self, params, text_generator):
        super(RAGTrack, self).__init__(params)
        network = build_ragtrack(params.cfg, training=False)

        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print('Load pretrained model from: ' + self.params.checkpoint)
        self.text_generator = text_generator
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM(mean= self.cfg.DATA.MEAN, std = self.cfg.DATA.STD)
        self.state = None
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS
        self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD

        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False
        self.debug = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.sequence_id = RAGTrack._sequence_id_counter
        RAGTrack._sequence_id_counter += 1

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_dict = [template] * self.num_template

        self.box_mask_z = None
        self.track_query_before = None
        self.text_label = info['text_label']
        self.first_text_label = info['first_text_label']

        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_dict = [search]
            if self.track_query_before != None:
                out_dict = self.network.forward(
                    template= self.z_dict ,
                    search=x_dict, ce_template_mask=self.box_mask_z, track_query_before=self.track_query_before,text_label=self.text_label )
            else:
                out_dict = self.network.forward(
                    template= self.z_dict ,
                    search=x_dict, ce_template_mask=self.box_mask_z, text_label=self.text_label )

        self.track_query_before  = out_dict[0]['track_query_before']

        pred_score_map = out_dict[0]['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict[0]['size_map'], out_dict[0]['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        conf_score = None
        if self.num_template > 1:
            conf_score, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):

                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)
                self.z_patch_arr = z_patch_arr
                template = self.preprocessor.process(z_patch_arr)
                self.z_dict.append(template)
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)
                self.text_label = self.text_generator.generate_descriptions(self.first_text_label, image, *self.state)

        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            if conf_score != None:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":conf_score.cpu().numpy()[0][0]}
            else:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":None
                        }
        else:
            if conf_score != None:
                return {"target_bbox": self.state,
                        "best_score":conf_score.cpu().numpy()[0][0]
                        }
            else:
                return {"target_bbox": self.state,
                        "best_score":None
                        }


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return RAGTrack
