#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/tmi2022/results/feats_k_4.out
#SBATCH --error=/home/ofourkioti/Projects/tmi2022/results/error.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/tmi2022/feature_extractor/

#python build_graphs.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/graphs/"
#python build_graphs.py --weights "model.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA-LUNG/tiles/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA-LUNG/graphs"

#python compute_feats_gtp.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/colon_patches/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/graphs/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/segment_dataset/
<<<<<<< HEAD
#python compute_feats_gtp.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/gtp_features/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/images/
python compute_feats_res.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/datasets/camelyon_data/size_256/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/feats_k_4/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon16/
=======
#python compute_feats_gtp.py --weights "DDSMIL_extractors/tcga_lung/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/gtp_features/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/TCGA_LUNG/
python compute_feats_res.py --weights "DSMIL_extractors/tcga_lung/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/feats/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/TCGA_LUNG/
>>>>>>> 8d9cfac (compute tcga feats)
#python compute_feats_res.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/ds_feats_k4/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/images/
