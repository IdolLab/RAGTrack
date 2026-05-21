import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
import random
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.ragtrack import RAGTrack
import lib.test.parameter.ragtrack as rgbt_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
from Qwen_VL.qwen_generate import TextGenerator
text_generator = TextGenerator(model_path="/home/sqh/lihao/RAGTrack/Qwen_VL/qwen2.5-vl-3b")

def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

        desc_files = [
            os.path.join(seq_path, 'visible_description.txt')
        ]

        text_label = []
        for fp in desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    text_label.append((f.read().strip(),))
            else:
                text_label.append((None,))
                print(f'Missing description file: {fp}')

        first_desc_files = [
            os.path.join(seq_path, 'text.txt')
        ]

        first_text_label = []
        for fp in first_desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    first_text_label.append((f.read().strip(),))  # 注意逗号：单元素元组
            else:
                first_text_label.append((None,))  # 同样包装成单元素元组
                print(f'Missing description file: {fp}')

    elif set_type == 'RGBT210':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')

        desc_files = [
            os.path.join(seq_path, 'visible_description.txt')
        ]

        text_label = []
        for fp in desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    text_label.append((f.read().strip(),))
            else:
                text_label.append((None,))
                print(f'Missing description file: {fp}')

        first_desc_files = [
            os.path.join(seq_path, 'text.txt')
        ]

        first_text_label = []
        for fp in first_desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    first_text_label.append((f.read().strip(),))  # 注意逗号：单元素元组
            else:
                first_text_label.append((None,))  # 同样包装成单元素元组
                print(f'Missing description file: {fp}')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1].lower() in ['.png', '.bmp']])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1].lower() in ['.png', '.bmp']])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        desc_files = [
            os.path.join(seq_path, 'visible_description.txt')
        ]

        text_label = []
        for fp in desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    text_label.append((f.read().strip(),))
            else:
                text_label.append((None,))
                print(f'Missing description file: {fp}')

        first_desc_files = [
            os.path.join(seq_path, 'text.txt')
        ]

        first_text_label = []
        for fp in first_desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    first_text_label.append((f.read().strip(),))  # 注意逗号：单元素元组
            else:
                first_text_label.append((None,))  # 同样包装成单元素元组
                print(f'Missing description file: {fp}')
    
    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

        desc_files = [
            os.path.join(seq_path, 'visible_description.txt')
        ]

        text_label = []
        for fp in desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    text_label.append((f.read().strip(),))
            else:
                text_label.append((None,))
                print(f'Missing description file: {fp}')

        first_desc_files = [
            os.path.join(seq_path, 'text.txt')
        ]

        first_text_label = []
        for fp in first_desc_files:
            if os.path.isfile(fp):
                with open(fp, encoding='utf-8') as f:
                    first_text_label.append((f.read().strip(),))  # 注意逗号：单元素元组
            else:
                first_text_label.append((None,))  # 同样包装成单元素元组
                print(f'Missing description file: {fp}')

    return RGB_img_list, T_img_list, RGB_gt, T_gt, text_label, first_text_label


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='prompt', dropped_seqs=None):
    if dropped_seqs is None:
        dropped_seqs = set()

    seq_txt = seq_name
        
    save_name = '{}'.format(yaml_name)
    save_path = f'./RGBT_workspace/results/{dataset_name}/{save_name}_{epoch}/' + seq_txt + '.txt'
    save_folder = f'./RGBT_workspace/results/{dataset_name}/{save_name}_{epoch}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
        
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    if script_name == 'ragtrack':
        params = rgbt_params.parameters(yaml_name, epoch)
        ragtrack = RAGTrack(params, text_generator)
        tracker = RAGTRACK_RGBT(tracker=ragtrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt, text_label, first_text_label = genConfig(seq_path, dataset_name)

    if seq_name in dropped_seqs:
        text_label = [("",)]
        print(f'  >>> Dropping text label for ablation study')
    
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist(), text_label, first_text_label)
        elif frame_idx > 0:
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            region, confidence = tracker.track(image)
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class RAGTRACK_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region, text_label, first_text_label):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np),
                    'text_label': text_label,
                     'first_text_label': first_text_label}
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='ragtrack', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='RAGTrack', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset (GTOT,RGBT210,RGBT234,LasHeR).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=19, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='sequential', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--text_drop_ratio', default=0.0, type=float, help='Ratio of sequences to drop text labels for ablation study (0.0 to 1.0)')

    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name

    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/home/sqh/lihao/RAGTrack/data/RGBTdatasets/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'RGBT210':
        seq_home = '/home/sqh/lihao/RAGTrack/data/RGBTdatasets/RGBT210'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = '/home/sqh/lihao/RAGTrack/data/RGBTdatasets/RGBT234'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/home/sqh/lihao/RAGTrack/data/RGBTdatasets/LasHeR/test'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    random.seed(42)
    seq_list.sort()
    num_drop = int(len(seq_list) * args.text_drop_ratio)
    dropped_seqs = set(random.sample(seq_list, num_drop)) if num_drop > 0 else set()
    
    if num_drop > 0:
        print(f'========== Ablation Study: Dropping text labels for {num_drop}/{len(seq_list)} sequences ==========')
        print(f'Dropped sequences: {dropped_seqs}')

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, dropped_seqs) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, dropped_seqs) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")

