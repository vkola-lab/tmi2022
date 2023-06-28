#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=76:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/tile.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/error.err
#SBATCH --partition=smp

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate card
cd /home/ofourkioti/Projects/tmi2022/

python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 20 -o  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/ "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_slides/slides/*.tif"

#python src/tile_WSI.py -s 512 -e 0 -B 50 -M 20 -o  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA_LUNG/tiles/ "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/TCGA_data/*.svs"

