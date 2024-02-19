## Search and Pruning (SnP) Baseline for Training Set Search


The code is based on Yue's SnP framework for training set search, the paper 'Large-scale Training Data Search for Object Re-identification' in CVPR2023.


<!-- ## Requirements

Please refer to [requirment](https://github.com/yorkeyao/DataCV2024/tree/main/task_model) of the task model. Addtionally, we need:

- Sklearn
- Scipy 1.2.1 -->

## Datasets Preparation

Please refer to [dataset download page](https://github.com/yorkeyao/DataCV2024/tree/main) for datasets. After all data is ready, please modify the paths for dataset in 'trainingset_search_detection_vehicle.py'

## Running example 

For running such process, when Market is used as target, we can seach a training set with 8000 images using the command below:

```python
python trainingset_search_detection_vehicle.py --target 'region100' \
--select_method 'SnP' --c_num 50 \
--result_dir 'main_results/sample_data_detection_vehicle_region100/' \
--n_num 8000 \
--output_data '/data/detection_data/trainingset_search/SnP_region100_vehicle_8000_random_c_num50.json'  
```
Please modify the output json file to a suitable place. After a training set is searched. Please use the [task model](https://github.com/yorkeyao/DataCV2024/tree/main/task_model) to get a prediction for evaluation.  


## Citation

If you find this code useful, please kindly cite:

```
@article{yao2023large,
  title={Large-scale Training Data Search for Object Re-identification},
  author={Yao, Yue and Lei, Huan and Gedeon, Tom and Zheng, Liang},
  journal={arXiv preprint arXiv:2303.16186},
  year={2023}
}
```

If you have any question, feel free to contact yue.yao@anu.edu.au
