#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=50:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/compute_features_camelyon-dsmil.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/compute_features_camelyon.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate card
cd /home/ofourkioti/Projects/tmi2022/feature_extractor


python compute_feats_res.py --weights "DSMIL_extractors/20x/model-v1.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/datasets/camelyon_data/size_256/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/dsmil-feats/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_slides/slides
