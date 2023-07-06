#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/compute_feats_cam_16_gtp.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/compute_feats_cam_16_gtp.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate dl_torch
cd /home/ofourkioti/Projects/tmi2022/feature_extractor

#python build_graphs.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/graphs/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_slides/slides/
#python build_graphs.py --weights "model.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA-LUNG/tiles/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA-LUNG/graphs"
#python compute_feats_res.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/feats/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/images/
python compute_feats_gtp.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/datasets/camelyon_data/size_256/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/gtp_features/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon_slides/

#python compute_feats_gtp.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/gtp_features/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/images/
