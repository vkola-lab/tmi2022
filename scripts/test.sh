export CUDA_VISIBLE_DEVICES=0
python -W ignore main.py \
--n_class 3 \
--data_path "/scratch2/zheng/cptac_data/" \
--val_set "cptac_lung_val1.txt" \
--model_path "/scratch2/zheng/kidney_fibrosis_patch_based/Github/deep_globe/saved_models/" \
--log_path "/scratch2/zheng/kidney_fibrosis_patch_based/GLNet/deep_globe/runs/" \
--task_name "GraphCAM" \
--batch_size 1 \
--test \
--log_interval_local 6 \
--resume "/scratch2/zheng/kidney_fibrosis_patch_based/Github/deep_globe/saved_models_val1/GraphCAM.pth"