#!/bin/bash
#SBATCH --job-name=datatransfer_test
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/datatransfer_test.txt
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/datatransfer_test.err
#SBATCH --partition=data-transfer
#SBATCH --ntasks=1
#SBATCH --time=100:00:00

#srun rsync -avP   /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/

#srun rsync -avP /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/
srun rsync -avP /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/graphcam/*  /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/graphcam/
