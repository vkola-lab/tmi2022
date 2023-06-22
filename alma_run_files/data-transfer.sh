#!/bin/bash
#SBATCH --job-name=datatransfer_test
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/datatransfer_test.txt
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/datatransfer_test.err
#SBATCH --partition=data-transfer
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

#srun rsync -avP   /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/

srun rsync -avP /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA-LUNG/*  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/
#srun rsync -avP    /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/Nature-2019-patches/* /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/Nature-2019-patches/
#srun rsync -avP   /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/lipos_flat/* /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/lipos/f lat/