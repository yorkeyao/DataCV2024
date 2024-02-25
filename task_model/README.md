# Overview

Our task model is based on [mmdetection](https://github.com/open-mmlab/mmdetection). 

# Modificantion of Data Path

You will need to modify the link to dataset, please modify the data path in 'task_model/configs_tss/_base_/datasets/custom_tss.py'. Please note the annotation file for testA is attached in this repo and only contains images.  

# Training and testing

You may perform training with multiple GPUs. The following command is used to perform training on 8 GPUs. 

```python
bash tools/dist_train.sh configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py 8
```

or training with a single GPU

```python
python tools/train.py configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py 
```

This will create a checkpoint in './work_dirs'. You may create a coco format submission using 

```python
python tools/test.py \
    configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py \
    work_dirs/retinanet_r50_fpn_1x_custom_tss_car/latest.pth \
    --format-only --options "jsonfile_prefix=./"
```

It will output ".bbox.json" at "./", this is the output we need. Please have it renamed to 'answer.txt' and zip it for the evaluation server. 

