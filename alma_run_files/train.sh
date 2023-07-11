#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/train_cam16.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/train.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate my_mamba_environment dl_torch
cd /home/ofourkioti/Projects/tmi2022/

for i in {0..4};
do CUDA_VISIBLE_DEVICES=0 python main.py --n_class 2 --data_path "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/gtp_features/" \
--train_set "cam_16_splits/train_${i}.txt" --val_set "cam_16_splits/val_${i}.txt" --model_path "graph_transformer/saved_models/" \
--log_path "graph_transformer/runs/" \
--task_name "camelyon16_fold_${i}" \
--batch_size 8 \
--train \
--log_interval_local 6
done