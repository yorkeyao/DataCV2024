# DataCV2024: The 2nd CVPR DataCV Challenge

Welcome to DataCV2024!

This is the development kit repository for [the 2nd CVPR DataCV Challenge](https://sites.google.com/view/vdu-cvpr24/competition/). Here you can find details on how to download datasets, run baseline models, and evaluate the performance of your model. Evaluation can be performed remotely on the [CodeLab](https://codalab.lisn.upsaclay.fr/competitions/17688) evaluation server. Please also refer to the main website for competition details, rules, and dates.

Have fun!


--------------------------------------------------------------------

## Overview 
This year’s challenge focuses on **Training set search for object detection**. We consider a scenario where we have access to the target domain, but cannot afford on-the-fly training data annotation, and instead would like to construct an alternative training set from a large-scale data pool such that a competitive model can be obtained. Specifically, we focuses on vehicle detection. We newly introudce Regin100 as our target and reuse existing 6 well-known benchmarks as source pool. 

The competition will take place during Feb -- Apr 2024, and the top-performing teams will be invited to present their results at the workshop at [CVPR 2024](https://sites.google.com/view/vdu-cvpr24/home) in Jun, Seattle.

## Challenge Data 

Source pool: [[GoogleDrive]](https://drive.google.com/file/d/10kRIfJSxOdF84WMh9AR63YJDk07LPszo/view?usp=drive_link) and [[BaiduYun]](https://pan.baidu.com/s/1NeLvKAhrHgXn_Zul2VEfHw) (pwd: jp2x)

Target: [[GoogleDrive]](https://drive.google.com/file/d/1u9AfVxQpFTJkzm50Wfvr6LJhP9b2Dq1J/view?usp=sharing) and [[BaiduYun]](https://pan.baidu.com/s/1XZypSQsiVjjWfH9FFKerjw?pwd=yqt5) (pwd: yqt5) 


![fig1](https://github.com/yorkeyao/DataCV2024/blob/main/images/write.jpg)  
<!-- ![enter image description here](https://github.com/sxzrt/The-PersonX-dataset/raw/master/images/logo1.jpg) -->


The target dataset: Region100 benchmark consists of footage captured by 100 static cameras from various regions in the real world.
For videos from each different region, the first 70% is used for model training, while the remaining 30% is designated for validation and testing.

The target domain consists of real-world images from 100 regions in the world. In total, there are 21871 images. We have provided region camera index information for both source and target training sets. 
 - For the training set, it contains 15,368 images.
 - For the test A set, it contains 2,134 images.
 - For the test B set, it contains 4,368 images. The test B set will be researved for final ranking. 

The source pool comprises datasets from six existing sources: ADE, BDD, Cityscapes, COCO, VOC, Detrac, and KITTI. We have standardized the labeling format to match that of COCO, and filtered out labels that are not related to vehicles. In total, the collection contains 176,491 images. 

The challenge dataset split is organized as follows: 
```
├── Region100/
(Source pool from existing benchmarks)
│   ├── source_pool/              /* source training images
│       ├── voc_train/                    
│           ├── VOC2007/
|           ├── VOC2012
│       ├── kitti_train/                    
│           ├── 000000.png
|           ├── 000001.png
|           ...
│       ├── coco_train/                    
│           ├── 000000000009.jpg
|           ├── 000000000025.jpg
|           ...
│       ├── cityscapes_train/                    
│           ├── aachen_000000_000019_leftImg8bit.jpg
|           ├── aachen_000001_000019_leftImg8bit.jpg
|           ...
│       ├── bdd_train/                    
│           ├── 0a0a0b1a-7c39d841.jpg
|           ├── 0a0b16e2-93f8c456.jpg
|           ...
│       ├── ade_train/                    
│           ├── ADE_train_00000001.jpg
|           ├── ADE_train_00000002.jpg
|           ...
|       ├── detrac_train/  
│       ├── voc_annotation.json
│       ├── kitti_annotation.json
│       ├── coco_annotation.json
│       ├── cityscapes_annotation.json
│       ├── bdd_annotation.json
│       ├── ade_annotation.json 
(Target region100 dataset collected from real world)
│   ├── train/                    /* training set
│       ├── 001/
|           ├── 001_000001.jpg
|           ├── 001_000002.jpg
|           ...
│       ├── 002/
|           ├── 002_000001.jpg
|           ├── 002_000002.jpg
|           ...
│   ├── testA/                    /* test A set
│       ├── 001_000001.jpg
│       ├── 001_000002.jpg
│       ├── 001_000003.jpg
│       ├── 001_000004.jpg
|           ...
│   ├── testB/                     /* test B set (not released yet)
│       ├── 001_000001.jpg
│       ├── 001_000002.jpg
│       ├── 001_000003.jpg
│       ├── 001_000004.jpg
|           ...
```

The aim of this challenge is to find no more than 8000 images from source pool to train a competent model for our region100 target. [A baseline algorithm](https://github.com/yorkeyao/DataCV2024/tree/main/SnP_detection) for training set search has been provided. 

### Naming Rule of the Images
The first three digits correspond to the region code, and the following six digits correspond to the image number within the region. For example, 001_000001.jpg is the first image captured from the first region camera. "001" is the region code, and "000001" is the image counting number.

By downloading these datasets you agree to the following terms:

### Terms of Use
- The data is to be utilized solely for research and educational purposes that do not seek profit. 
- The distribution of images obtained from the data is prohibited. 
- The providers of the data disclaim all guarantees, including but not limited to those concerning the violation of rights or the data's suitability for a specific use. 
- Users assume all responsibility for their application of the data. 

### Detection Model

Unlike traditional challenges where the training set is fixed, allowing participants to optimize their model or algorithm, in our 'training set search' challenge, we keep the model or algorithm fixed and allow participants to optimize the training set. We fix our task model as RetinaNet, which is available on [this link](https://github.com/yorkeyao/DataCV2024/tree/main/task_model). 

## Feedback and Help
If you find any bugs please [open an issue](https://github.com/yorkeyao/datacv2024/issues/new).

## References

```
@inproceedings{yao2023large,
  title={Large-scale Training Data Search for Object Re-identification},
  author={Yao, Yue and Lei, Huan and Gedeon, Tom and Zheng, Liang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15568--15578},
  year={2023}
}
```
