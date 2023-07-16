#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/train.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/train.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/tmi2022/

EXPORT CUDA_VISIBLE_DEVICES=0
#for i in {0..4};
python main.py --n_class 2 --data_path "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/" \
--train_set "cam_16_splits/train_4.txt" --val_set "cam_16_splits/val_4.txt" --model_path "graph_transformer/saved_models/" \
--log_path "graph_transformer/runs/" \
--task_name "train_cam16_tile_features_4" \
--batch_size 2 \
--train \
--log_interval_local 5