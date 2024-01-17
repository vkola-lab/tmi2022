"""Dataset class for the graph classification task."""

import os
from typing import Any

import torch
import torch.utils.data as data
# import numpy as np
from PIL import ImageFile
# from torchvision import transforms
# import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True


# def collate_features(batch):
#     img = torch.cat([item[0] for item in batch], dim=0)
#     coords = np.vstack([item[1] for item in batch])
#     return [img, coords]


# def eval_transforms(pretrained=False):
#     if pretrained:
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)

#     else:
#         mean = (0.5, 0.5, 0.5)
#         std = (0.5, 0.5, 0.5)

#     trnsfrms_val = transforms.Compose(
#         [
#             transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)
#         ]
#     )

#     return trnsfrms_val


class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root: str, ids: list[str], target_patch_size=-1, site: str | None = 'LUAD', classdict: dict[str, int] | None = None):
        super(GraphDataset, self).__init__()
        """
        Args:

        root (string):  directory with all the input images.
        ids (list[str]):  names and labels of every graph in the dataset, formatted as 'site/file_name\tlabel'.
        target_patch_size (int):
        site (str, optional):  name of the tumor site, if specified. Defaults to 'LUAD'.
        classdict (dict[str, int], optional):  dictionary mapping class names to class indices, if specified. Defaults to None.
        """
        self.root = root
        self.ids = ids
        #self.target_patch_size = target_patch_size

        if site in {'LUAD', 'LSCC'}:
            self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}        #
        elif site == 'NLST':
            self.classdict = {'normal': 0, 'tumor': 1}        #
        elif site == 'TCGA':
            self.classdict = {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}
        elif site is None:
            self.classdict = None
        else:
            raise ValueError('Site not recognized: {}'.format(site))
        self.site = site

        self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        site, file_name = file_name.split('/')

        # Default
        if self.site is None:
            file_path = self.root

        # if site =='CCRCC':
        #     file_path = self.root + 'CPTAC_CCRCC_features/simclr_files'
        if site =='LUAD' or site =='LSCC':
            site = 'LUNG'
            file_path = self.root + 'CPTAC_{}_features/simclr_files'.format(site)       #_pre# with # rushin

        # For NLST only
        elif site =='NLST':
            file_path = self.root + 'NLST_Lung_features/simclr_files'

        # For TCGA only
        elif site =='TCGA':
            file_name = info.split('\t')[0]
            _, file_name = file_name.split('/')
            file_path = self.root + 'TCGA_LUNG_features/simclr_files'       #_resnet_with
        
        else:
            raise RuntimeError('Site not recognized: {}'.format(site))

        if self.classdict is not None:
            sample['label'] = self.classdict[label]
        else:
            try:
                sample['label'] = int(label)
            except ValueError:
                raise ValueError(f'If no classdict is provided, labels must be integers. Got: {label}')
        sample['id'] = file_name

        feature_path = os.path.join(graph_path, graph_name, 'features.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            raise FileNotFoundError(f'features.pt for {graph_name} doesn\'t exist')

        adj_s_path = os.path.join(graph_path, graph_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else:
            raise FileNotFoundError(f'adj_s.pt for {graph_name} doesn\'t exist')

        sample['image'] = features
        sample['adj_s'] = adj_s

        return sample

    def __len__(self):
        return len(self.ids)
