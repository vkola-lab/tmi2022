#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/get_graphcam.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/get_graphcam.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/tmi2022/


for i in {0..4};
do export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 2 \
--data_path '/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy' \
--val_set "colon_splits/test_${i}.txt" \
--model_path "graph_transformer/saved_models/" \
--log_path "graph_transformer/runs/" \
--task_name "colon_${i}"  \
--batch_size 1 \
--test \
--log_interval_local 5 \
--resume "graph_transformer/saved_models/colon_${i}.pth" \
--graphcam
done

#for i in {0..4};
#do export CUDA_VISIBLE_DEVICES=0
#python main.py \
#--n_class 2 \
#--data_path '/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16' \
#--val_set "cam_16_splits/test_${i}.txt" \
#--model_path "graph_transformer/saved_models/" \
#--log_path "graph_transformer/runs/" \
#--task_name "train_cam16_tile_features_${i}"  \
#--batch_size 1 \
#--test \
#--log_interval_local 5 \
#--resume "graph_transformer/saved_models/train_cam16_tile_features_${i}..pth" \
#--graphcam
#done