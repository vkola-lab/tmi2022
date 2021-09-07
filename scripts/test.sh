export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 3 \
--data_path "path_to_graph_data" \
--val_set "test_set.txt" \
--model_path "../graph_transformer/saved_models/" \
--log_path "../graph_transformer/runs/" \
--task_name "GraphCAM" \
--batch_size 1 \
--test \
--log_interval_local 6 \
--resume "../graph_transformer/saved_models/GraphCAM.pth"
