export CUDA_VISIBLE_DEVICES=1
python -W ignore main.py \
--n_class 3 \
--data_path "/scratch2/zheng/cptac_data/" \
--train_set "cptac_lung_train1.txt" \
--val_set "cptac_lung_val1.txt" \
--model_path "/scratch2/zheng/kidney_fibrosis_patch_based/Github/deep_globe/saved_models_val1/" \
--log_path "/scratch2/zheng/kidney_fibrosis_patch_based/Github/deep_globe/val1/" \
--task_name "GraphCAM" \
--batch_size 8 \
--train \
--log_interval_local 6 \