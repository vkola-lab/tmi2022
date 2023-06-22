import cl as cl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        img = img.resize((224, 224))
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def save_coords(txt_file, csv_file_path):
    for path in csv_file_path:
        x, y = path.split('/')[-1].split('.')[0].split('_')
        txt_file.writelines(str(x) + '\t' + str(y) + '\n')
    txt_file.close()

def adj_matrix(csv_file_path, output):
    total = len(csv_file_path)
    adj_s = np.zeros((total, total))

    for i in range(total-1):
        path_i = csv_file_path[i]
        x_i, y_i = path_i.split('/')[-1].split('.')[0].split('_')
        for j in range(i+1, total):
            # sptial 
            path_j = csv_file_path[j]
            x_j, y_j = path_j.split('/')[-1].split('.')[0].split('_')
            if abs(int(x_i)-int(x_j)) <=1 and abs(int(y_i)-int(y_j)) <= 1:
                adj_s[i][j] = 1
                adj_s[j][i] = 1

    adj_s = torch.from_numpy(adj_s)
    adj_s = adj_s.cuda()

    return adj_s

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(args, bags_list, i_classifier, save_path=None, whole_slide_path=None):
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if  args.magnification == '20x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpeg'))
            file_name = bags_list[i].split('/')[-3].split('_')[0]
        if args.magnification == '5x' or args.magnification == '10x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))

        dataloader, bag_size = bag_dataset(args, csv_file_path)
        print('{} files to be processed: {}'.format(len(csv_file_path), file_name))

        if os.path.isdir(os.path.join(save_path, 'simclr_files', file_name)) or len(csv_file_path) < 1:
            print('alreday exists')
            continue
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                #feats = feats.cpu().numpy()
                feats_list.extend(feats)
        
        os.makedirs(os.path.join(save_path, 'simclr_files', file_name), exist_ok=True)

        txt_file = open(os.path.join(save_path, 'simclr_files', file_name, 'c_idx.txt'), "w+")
        save_coords(txt_file, csv_file_path)
        # save node features
        output = torch.stack(feats_list, dim=0).cuda()
        torch.save(output, os.path.join(save_path, 'simclr_files', file_name, 'features.pt'))
        # save adjacent matrix
        adj_s = adj_matrix(csv_file_path, output)
        torch.save(adj_s, os.path.join(save_path, 'simclr_files', file_name, 'adj_s.pt'))

        print('\r Computed: {}/{}'.format(i+1, num_bags))
        

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--num_feats', default=512, type=int, help='Feature size')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--dataset', default=None, type=str, help='path to patches')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone')
    parser.add_argument('--magnification', default='20x', type=str, help='Magnification to compute features')
    parser.add_argument('--weights', default=None, type=str, help='path to the pretrained weights')
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    args = parser.parse_args()
    
    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = cl.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
    
    # load feature extractor
    if args.weights is None:
        print('No feature extractor')
        return
    state_dict_weights = torch.load(args.weights)
    state_dict_init = i_classifier.state_dict()
    new_state_dict = OrderedDict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    i_classifier.load_state_dict(new_state_dict, strict=False)
 
    os.makedirs(args.output, exist_ok=True)
    bags_list = glob.glob(args.dataset)
    compute_feats(args, bags_list, i_classifier, args.output)
    
if __name__ == '__main__':
    main()
