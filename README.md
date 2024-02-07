# DataCV2024: The 2nd CVPR DataCV Challenge

Welcome to DataCV2024!

This is the development kit repository for [the 2nd CVPR DataCV Challenge](https://sites.google.com/view/vdu-cvpr24/competition/). Here you can find details on how to download datasets, run baseline models (TODO), and evaluate the performance of your model (TODO). Evaluation can be performed remotely on the CodeLab evaluation server. Please also refer to the main website for competition details, rules, and dates.

Have fun!


--------------------------------------------------------------------

## Overview 
This year’s challenge focuses on **Training set search for object detection**. We consider a scenario where we have access to the target domain, but cannot afford on-the-fly training data annotation, and instead would like to construct an alternative training set from a large-scale data pool such that a competitive model can be obtained. Specifically, we focuses on person and vehicle detection. We newly introudce Regin100 as our target and reuse existing 6 well-known benchmark as source pool. 

The competition will take place during Feb -- Apr 2024, and the top-performing teams will be invited to present their results at the workshop at [CVPR 2024](https://sites.google.com/view/vdu-cvpr24/home) in Jun, Seattle.

## Challenge Data 

Source pool: [[GoogleDrive]](https://drive.google.com/file/d/1n5k9V0YOO6DWuMgO5xm5TBc2DRLAXdEq/view?usp=sharing) and [[BaiduYun]](https://1drv.ms/u/s!AhjrHmxemkOga91UXOVXsVZJqTg?e=kbE1CC)

Target: [[GoogleDrive]](https://drive.google.com/file/d/1u9AfVxQpFTJkzm50Wfvr6LJhP9b2Dq1J/view?usp=sharing) and [[BaiduYun]](https://1drv.ms/u/s!AhjrHmxemkOga91UXOVXsVZJqTg?e=kbE1CC)


![fig1](https://github.com/yorkeyao/datacv24/blob/main/images/write.jpg)  
<!-- ![enter image description here](https://github.com/sxzrt/The-PersonX-dataset/raw/master/images/logo1.jpg) -->


The target dataset: Region100 benchmark consists of footage captured by 100 static cameras from various regions in the real world.
For videos from each different region, the first 70% is used for model training, while the remaining 30% is designated for validation and testing.

The target domain consists of real-world images from 100 regions in the world. In total, there are 21871 images. We have provided region camera index information for both source and target training sets. 
 - For the training set, it contains 15,369 images.
 - For the test A set, it contains 2,134 images.
 - For the test B set, it contains 4,368 images. The test B set will be researved for final ranking. 

The source pool comprises datasets from six existing sources: ADE, BDD, Cityscapes, COCO, VOC, and KITTI. We have standardized the labeling format to match that of COCO, and filtered out labels that are not related to persons or vehicles. In total, the collection contains 161,789 images. 

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

### Naming Rule of the images
The first three digits correspond to the region code, and the following six digits correspond to the image number within the region. For example, 001_000001.jpg is the first image captured from the first region camera. "001" is the region code, and "000001" is the image counting number.

By downloading these datasets you agree to the following terms:

### Terms of Use
- The data is to be utilized solely for research and educational purposes that do not seek profit. 
- The distribution of images obtained from the data is prohibited. 
- The providers of the data disclaim all guarantees, including but not limited to those concerning the violation of rights or the data's suitability for a specific use. 
- Users assume all responsibility for their application of the data. 

### Detection Model

Unlike traditional challenges where the training set is fixed, allowing participants to optimize their model or algorithm, in our 'training set search' challenge, we keep the model or algorithm fixed and allow participants to optimize the training set. We fix our task model as RetinaNet, which is available on [mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/retinanet). 

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