import os
import feat_stas.SnP_detection
import argparse
import numpy as np
import torch
import random
from scipy.special import softmax

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--logs_dir', type=str, metavar='PATH', default='sample_data/log.txt')
parser.add_argument('--result_dir', type=str, metavar='PATH', default='sample_data_detection/')
parser.add_argument('--use_camera', action='store_true', help='use use_camera')
parser.add_argument('--random_sample', action='store_true', help='random sample')
parser.add_argument('--c_num', default=100, type=int, help='number of cluster')
parser.add_argument('--select_method', type=str, default='SnP', choices=['greedy', 'random', 'SnP'], help='how to sample')
parser.add_argument('--n_num', default=8000, type=int, help='number of ids')
parser.add_argument('--no_sample', action='store_true', help='do not perform sample')
parser.add_argument('--cuda', action='store_true', help='whether cuda is enabled')
parser.add_argument('--target', type=str, default='exdark', choices=['region100'], help='select which target')
parser.add_argument('--FD_model', type=str, default='inception', choices=['inception', 'posenet'],
                    help='model to calculate FD distance')
parser.add_argument('--output_data', type=str, metavar='PATH', default='/data/detection_data/searched.json')
parser.add_argument('--seed', default=0, type=int, help='number of cluster')

opt = parser.parse_args()
logs_dir=opt.logs_dir
result_dir=opt.result_dir

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
random.seed(opt.seed) 

data_dict = {
        'ade': '/data2/source_pool/ade_train/',
        'bdd': '/data2/source_pool/bdd_train/', 
        'cityscape': '/data2/source_pool/cityscapes_train/', 
        'coco': '/data2/source_pool/coco_train/',
        'detrac': '/data2/source_pool/detrac_train/',
        'kitti': '/data2/source_pool/kitti_train/',
        'voc': '/data2/source_pool/voc_train/',
        'region100': '/data2/region_100/train/'
        }

annotation_dict = {
        'ade': '/data2/source_pool/ade_annotation.json',
        'bdd': '/data2/source_pool/bdd_annotation.json', 
        'cityscape': '/data2/source_pool/cityscapes_annotation.json', 
        'coco': '/data2/source_pool/coco_annotation.json',
        'detrac': '/data2/source_pool/detrac_annotation.json',
        'exdark': '/data2/source_pool/exdark_annotation.json',
        'kitti': '/data2/source_pool/kitti_annotation.json',
        'voc': '/data2/source_pool/voc_annotation.json'
        }

databse_id= ['ade', 'bdd', 'cityscape', 'coco', 'detrac', 'kitti', 'voc']

if opt.target == 'region100':
    target = data_dict['region100'] 

result_dir=opt.result_dir
c_num = opt.c_num

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

if os.path.isdir(opt.output_data):
    assert ("output dir has already exist")

sampled_data = feat_stas.SnP_detection.training_set_search(target, data_dict, annotation_dict, databse_id, opt, result_dir, c_num, version = "vehicle")


