from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--magnification', type=str, default='20x')
    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
