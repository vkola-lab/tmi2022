#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/vis_graphcam.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/vis_graphcam.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/tmi2022/

python src/vis_graphcam.py --path_file "cam_16_splits/test_0.txt"  --path_patches "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/" --path_WSI "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon16/" --path_graph "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/graphs/simclr_files/"
