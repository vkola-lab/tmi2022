#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=76:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/tile.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/error.err
#SBATCH --partition=smp

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/tmi2022/

python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 1 -o  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/tiles/ "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/images/*.tif"
#python src/tile_WSI.py -s 512 -e 0 -B 50 -M 20 -o  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA_LUNG/tiles/ "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/TCGA_data/*.svs"

