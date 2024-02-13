CUDA_VISIBLE_DEVICES=7 python trainingset_search_detection_vehicle.py --target 'region100' \
--select_method 'SnP' --c_num 100 \
--result_dir 'main_results/sample_data_detection_vehicle_region100/' \
--n_num 8000 \
--output_data '/data/detection_data/trainingset_search/SnP_region100_vehicle_8000_random_c_num100.json' 