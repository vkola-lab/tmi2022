import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]
    
def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, target_patch_size=-1):
        super(GraphDataset, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.ids = ids
        #self.target_patch_size = target_patch_size
        self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}        #
        #self.classdict = {'normal': 0, 'tumor': 1}        #
        #self.classdict = {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}
        self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index):
        sample = {}
        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        site, file_name = file_name.split('/')

        # if site =='CCRCC':
        #     file_path = self.root + 'CPTAC_CCRCC_features/simclr_files'
        if site =='LUAD' or site =='LSCC':
            site = 'LUNG'
        file_path = self.root + 'CPTAC_{}_features/simclr_files'.format(site)       #_pre# with # rushin

        # For NLST only
        if site =='NLST':
            file_path = self.root + 'NLST_Lung_features/simclr_files'

        # For TCGA only
        if site =='TCGA':
            file_name = info.split('\t')[0]
            _, file_name = file_name.split('/')
            file_path = self.root + 'TCGA_LUNG_features/simclr_files'       #_resnet_with

        sample['label'] = self.classdict[label]
        sample['id'] = file_name

        #feature_path = os.path.join(self.root, file_name, 'features.pt')
        feature_path = os.path.join(file_path, file_name, 'features.pt')

        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            features = torch.zeros(1, 512)

        #adj_s_path = os.path.join(self.root, file_name, 'adj_s.pt')
        adj_s_path = os.path.join(file_path, file_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else:
            print(adj_s_path + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])

        #features = features.unsqueeze(0)
        sample['image'] = features
        sample['adj_s'] = adj_s     #adj_s.to(torch.double)
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}

        return sample


    def __len__(self):
        return len(self.ids)