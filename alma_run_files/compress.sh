#!/bin/bash
#SBATCH --job-name=datatransfer_test
#SBATCH -o comp.out
#SBATCH -e comp.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=24


#srun rsync -avP   /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/

#srun rsync -avP /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/
#srun rsync -avP /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/*  /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/slides/

tar --use-compress-program="pigz -k -9 -p20 -l" -cf /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/tiles.tar.gz /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/tiles/

