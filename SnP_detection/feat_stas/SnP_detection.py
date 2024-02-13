#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torchvision
from k_means_constrained import KMeansConstrained

import numpy as np
import torch
from scipy import linalg
# from scipy.misc import imread
from matplotlib.pyplot import imread, imsave
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
from scipy import misc
import random
import re
from scipy.special import softmax
from collections import defaultdict
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mmdet.datasets.coco_car import CocoCarDataset
import json
import clip
import collections
from glob import glob
import os.path as osp
import h5py
import scipy.io
import threading
from PIL import Image
import copy
import pickle
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans
import time
import xml.dom.minidom as XD
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from feat_stas.dataloader import get_detection_data_vehicle
from feat_stas.models.inception import InceptionV3
from feat_stas.feat_extraction import get_activations, calculate_frechet_distance, calculate_activation_statistics

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='3', type=str,
                    help='GPU to use (leave blank for CPU only)')


def make_square(image, max_dim = 512):
    max_dim = max(np.shape(image)[0], np.shape(image)[1])
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image



def training_set_search(tpaths, data_dict, annotation_dict, dataset_id, opt, result_dir, c_num, version):
    """clustering the ids from different datasets and sampleing"""

    
    if version == 'vehicle':
        img_paths,  annotations,  dataset_ids, meta_dataset  = get_detection_data_vehicle (dataset_id, data_dict, annotation_dict)
    
    print (len (img_paths), len(annotations), len(dataset_ids), len (meta_dataset))

    if opt.FD_model == 'inception':
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    cuda = True
    model.cuda()
    batch_size=50

    # caculate the various, mu sigma of target ste
    print('=========== extracting feature of target traning set ===========')
    files = []
    if version == 'vehicle':
        region_ids = []
        if opt.target == 'region100':
            images = []
            for root, dirs, files_dir in os.walk(data_dict['region100']):
                for file in files_dir:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        regionID = int(file.split("_")[0])
                        region_ids.append (regionID)
                        files.append(os.path.join(root, file))

    if not os.path.exists(result_dir + '/target_feature.npy'):
        target_feature = get_activations(opt, files, model, batch_size, dims, cuda, verbose=False)
        np.save(result_dir + '/target_feature.npy', target_feature)
    else:
        target_feature = np.load(result_dir + '/target_feature.npy')
    m1 = np.mean(target_feature, axis=0)
    s1 = np.cov(target_feature, rowvar=False)
    sum_eigen_val1 = (s1.diagonal()).sum()

    # extracter feature for data pool
    if not os.path.exists(result_dir + '/feature_infer.npy'):
        print('=========== extracting feature of data pool ===========')
        model.eval()
        feature_infer = get_activations(opt, img_paths, model, batch_size, dims, cuda, verbose=False)
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        np.save(result_dir + '/feature_infer.npy', feature_infer)
    else:
        feature_infer = np.load(result_dir + '/feature_infer.npy')

    # clustering ids based on ids' mean feature
    if not os.path.exists(result_dir + '/label_cluster_'+str(c_num)+'_img.npy'):
        print('=========== clustering ===========')
        estimator = KMeans(n_clusters=c_num)
        # print (c_num, int(np.shape (feature_infer)[0] / c_num ))
        # estimator = KMeansConstrained(n_clusters=c_num, size_min=int(np.shape (feature_infer)[0] / c_num )-1, size_max=int(np.shape (feature_infer)[0] / c_num)+1)
        estimator.fit(feature_infer)
        label_pred = estimator.labels_
        np.save(result_dir + '/label_cluster_'+str(c_num)+'_img.npy',label_pred)
    else:
        label_pred = np.load(result_dir  + '/label_cluster_'+str(c_num)+'_img.npy')
    

    print('\r=========== caculating the fid and v_gap between T and C_k ===========')
    if not os.path.exists(result_dir + '/cluster_fid_div_by_'+str(c_num)+'_img.npy'):
        cluster_feature = []
        cluster_fid = []
        cluster_mmd = []
        cluster_var_gap = []
        cluster_div = []
        for k in tqdm(range(c_num)):

            initial_feature_infer = feature_infer[label_pred==k]
            cluster_feature.append(initial_feature_infer)
            
            mu = np.mean(initial_feature_infer, axis=0)
            sigma = np.cov(initial_feature_infer, rowvar=False)

            fea_corrcoef = np.corrcoef(initial_feature_infer)
            fea_corrcoef = np.ones(np.shape(fea_corrcoef)) - fea_corrcoef
            diversity_sum = np.sum(np.sum(fea_corrcoef)) - np.sum(np.diagonal(fea_corrcoef))
            current_div = diversity_sum / (np.shape (fea_corrcoef)[0] ** 2 - np.shape (fea_corrcoef)[0])

            # caculating variance
            current_var_gap = np.abs((sigma.diagonal()).sum() - sum_eigen_val1)

            current_fid = calculate_frechet_distance(m1, s1, mu, sigma)
            cluster_fid.append(current_fid)
            cluster_div.append(current_div)
            cluster_var_gap.append(current_var_gap)
        np.save(result_dir + '/cluster_fid_div_by_'+str(c_num)+'_img.npy', np.c_[np.array(cluster_fid), np.array(cluster_div)])
    else:
        cluster_fid_var=np.load(result_dir + '/cluster_fid_div_by_'+str(c_num)+'_img.npy')
        cluster_fid=cluster_fid_var[:,0]
        cluster_div=cluster_fid_var[:,1]


    cluster_fida = np.array(cluster_fid)
    score_fid = softmax(-cluster_fida)
    sample_rate = score_fid

    c_num_len = []
    id_score = []
    for kk in range(c_num):
        c_num_len_k = np.sum (label_pred == kk)
        c_num_len.append (c_num_len_k)

    for jj in range(len(label_pred)):
        id_score.append( sample_rate[label_pred[jj]] / c_num_len[label_pred[jj]])

    print (np.shape (id_score))
    # select a number index based on their FD score to the target domain 
    if opt.select_method == 'random':
        selected_data_ind = np.sort(np.random.choice(range(len(id_score)), opt.n_num, replace=False))

    if opt.select_method == 'greedy':
        selected_data_ind = np.argsort(id_score)[-opt.n_num:]
    
    if opt.select_method == 'SnP': 
        lowest_fd = float('inf')
        lowest_img_list = []
        if not os.path.exists(result_dir + '/domain_seletive_'+str(c_num)+'_img.npy'):
            cluster_rank = np.argsort(cluster_fida)
            current_list = []
            cluster_feature_aggressive = []
            for k in tqdm(cluster_rank):
                img_list = np.where (label_pred==k)[0]
                initial_feature_infer = feature_infer[label_pred==k]
                cluster_feature_aggressive.extend(initial_feature_infer)
                cluster_feature_aggressive_fixed = cluster_feature_aggressive
                target_feature_fixed = target_feature
                if len (cluster_feature_aggressive) > len (target_feature):
                    cluster_idx = np.random.choice(range(len (cluster_feature_aggressive)), len(target_feature), replace=False)
                    cluster_feature_aggressive_fixed = np.array([cluster_feature_aggressive[ii] for ii in cluster_idx])
                if len (cluster_feature_aggressive) < len (target_feature):
                    cluster_idx = np.random.choice(range(len(target_feature)), len (cluster_feature_aggressive), replace=False)
                    target_feature_fixed = target_feature[cluster_idx]
                mu = np.mean(cluster_feature_aggressive_fixed, axis=0)
                sigma = np.cov(cluster_feature_aggressive_fixed, rowvar=False)
                current_fid = calculate_frechet_distance(m1, s1, mu, sigma)
                current_list.extend (list (img_list))
                print (current_fid)
                if lowest_fd > current_fid:
                    lowest_fd = current_fid
                    lowest_img_list = copy.deepcopy(current_list)
            np.save(result_dir + '/domain_seletive_'+str(c_num)+'_img.npy', lowest_img_list)
        else:
            lowest_img_list = np.load(result_dir + '/domain_seletive_'+str(c_num)+'_img.npy')
        # print (len (lowest_img_list))
        selected_data_ind = lowest_img_list
    

    # print (len (selected_data_ind))
    if len (selected_data_ind) > opt.n_num:
        final_selected_img_ind = list(np.sort(np.random.choice(selected_data_ind, opt.n_num, replace=False)))
    else:
        final_selected_img_ind = selected_data_ind

    print('\r=========== building training set ===========')
    if not opt.no_sample:
        json_generate(final_selected_img_ind, meta_dataset, opt)

    # print (min(dataset_ids))
    # print (collections.Counter(dataset_ids[final_selected_img_ind]))
    
    result_feature = feature_infer[final_selected_img_ind]
    # print (np.shape (result_feature))

    mu = np.mean(result_feature, axis=0)
    sigma = np.cov(result_feature, rowvar=False)
    current_fid = calculate_frechet_distance(m1, s1, mu, sigma)

    # print (current_fid)

    idx = np.random.choice(range(len(result_feature)), 4000, replace=False)
    fea_corrcoef = np.corrcoef(result_feature[idx[:2000]], result_feature[idx[-2000:]])
    fea_corrcoef = np.ones(np.shape(fea_corrcoef)) - fea_corrcoef
    diversity_sum = np.sum(np.sum(fea_corrcoef))
    current_div = diversity_sum / (np.shape (fea_corrcoef)[0] ** 2)
    

    print('finished with a dataset has FD', current_fid, "and div", current_div)
    return selected_data_ind

INFO = {
    "description": "Cityscapes_Instance Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2020",
    "contributor": "Kevin_Jia",
    "date_created": "2020-1-23 19:19:19.123456"
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'car',
        'supercategory': 'car',
    },]

from pycococreatortools import pycococreatortools

def json_generate(selected_data_ind, meta_dataset, opt):

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    annotation_id = 0
    for image_id, idx in enumerate(selected_data_ind):
        image_path, anno, dataset_id = meta_dataset[idx]
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
                image_id, image_path, image.size)
        coco_output["images"].append(image_info)

        for annotation_info in anno:
            annotation_info['image_id'] = image_id
            annotation_info['id'] = annotation_id
            annotation_id = annotation_id + 1
            coco_output["annotations"].append(annotation_info)
    
    with open(opt.output_data, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)




