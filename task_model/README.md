# Overview

Our task model s based on [mmdetection](https://github.com/open-mmlab/mmdetection). 

# Modificantion of Data Path

You will need to modify the link to dataset, please mofity the data path in 'task_model/configs_tss/_base_/datasets/custom_tss.py'.  

# Training and testing

You may perform training with muliple GPUs. The following command is used to perform training on 8 GPUs. 

```python
bash tools/dist_train.sh configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py 8
```

or training with single GPU

```python
python tools/train.py configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py 
```

This will create checkpoint in './work_dirs'. You may create a coco formate submission using 

```python
python tools/test.py \
    configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py \
    work_dirs/retinanet_r50_fpn_1x_custom_tss_car/latest.pth \
    --format-only --options "jsonfile_prefix=./"
```

Please have it renamed to 'answer.txt' and zip it for evaluation server. 

