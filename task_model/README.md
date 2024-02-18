# Detection Environment

Our task model s based on [mmdetection](https://github.com/open-mmlab/mmdetection). 

To manage the python code, it is recommended to install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

For creating environment,

```python
conda create -n tss python=3.7 -y
conda activate tss
```

Besides, you will need to install pytorch 1.12, please modify cuda version according to the version installed in the system. 

```python
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

The install of mmcv library. 

```python
pip install -U openmim
mim install mmcv-full==1.7.0
pip install yapf==0.40.1
```

The install of mmdetction.

```python
cd mmdetection/
pip install -v -e . 
```

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
python tools/test_origin.py \
    configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py \
    work_dirs/retinanet_r50_fpn_1x_custom_tss_car/latest.pth \
    --format-only --options "jsonfile_prefix=./"
```

Please have it renamed to 'answer.txt' and zip it for evaluation server. 

