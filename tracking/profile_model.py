import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
import random, string

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ragtrack', choices=['ragtrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='RAGTrack', help='yaml configure file name')
    args = parser.parse_args()

    return args

def evaluate_vit(model, template, search, text_label):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search, text_label),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
    # keep_rate =[x for x in torch.linspace(0.7, 1, depth//4)][::-1]
    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, search, text_label)
            # if i == T_w-1:
            #     _ = model(template, search, text_label, text=True)
            # else:
            #     _ = model(template, search, text_label)
        start = time.time()
        for i in range(T_t):
            _ = model(template, search, text_label)
            # if i % 5 == 0:
            #     _ = model(template, search, text_label, text=True)
            # else:
            #     _ = model(template, search, text_label)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))

def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))

def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()

    '''update cfg'''

    
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    
    # work_path = os.getcwd() +'/'
    work_path =  '/home/sqh/lihao/RAGTrack/'
    config_module.update_config_from_file(work_path + yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "ragtrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_sttrack
        model = model_constructor(cfg, training=False)
        # get the template and search
        template = torch.randn(bs, 6, z_sz, z_sz)
        search = torch.randn(1, bs, 6, x_sz, x_sz)
        text_label = [
            (''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 15))),)
            for _ in range(1)
        ]
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)

        template = [template] * cfg.DATA.TEMPLATE.NUMBER

        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        if merge_layer <= 0:
            evaluate_vit(model, template, search, text_label)
        else:
            evaluate_vit_separate(model, template, search)

    else:
        raise NotImplementedError