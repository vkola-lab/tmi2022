#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=50:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/compute_features.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/error.err
#SBATCH --partition=smp

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate card
cd /home/ofourkioti/Projects/tmi2022/feature_extractor


python build_graphs.py --weights "model.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam_16/" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam_16/graphs/"
