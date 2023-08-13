import cl as cl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF
from datasets.dataset_h5 import  Whole_Slide_Bag_FP
import argparse, os, glob
from PIL import Image
from collections import OrderedDict
import h5py
from torchvision import transforms, utils, models
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
import openslide
from sklearn import preprocessing
import itertools

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

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=Compose([
                                         ToTensor()
                                     ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def generate_values_resnet(images, wsi_coords, dist="cosine"):
    """

    Parameters
    ----------
    images :  np.ndarray of size (numnber of patches,h,w,d) contatining the pathes of an image
    Idx    : indices of the closest neighbors of every image

    Returns
    -------
    a list of np.ndarrays, pairing every patch of an image with its closest neighbors

    """
    patch_distances = pairwise_distances(wsi_coords, metric='euclidean', n_jobs=1)
    neighbor_indices = np.argsort(patch_distances, axis=1)[:, :4]
    rows = np.asarray([[enum] * len(item) for enum, item in enumerate(neighbor_indices)]).ravel()
    columns = neighbor_indices.ravel()
    values = []
    coords = []
    for row, column in zip(rows, columns):
            m1 = np.expand_dims(images[int(row)], axis=0)
            m2 = np.expand_dims(images[int(column)], axis=0)
            # if (abs(int(wsi_coords[int(row)][0]) - int(wsi_coords[int(column)][0])))>=1024 and (abs(int(wsi_coords[int(row)][1]) - int(wsi_coords[int(column)][1])))>=1024 :
            #         value=np.inf
            # else:
            value = distance.cdist(m1.reshape(1, -1), m2.reshape(1, -1), dist)[0][0]
            values.append(value)
            coords.append((row, column))

    #values = np.reshape(values, (wsi_coords.shape[0], neighbor_indices.shape[1]))
    return np.array(coords), np.array(values), neighbor_indices

def adj_matrix(wsi_coords,wsi_feats):
    total = wsi_coords.shape[0]

    patch_distances = pairwise_distances(wsi_coords, metric='euclidean', n_jobs=1)
    neighbor_indices = np.argsort(patch_distances, axis=1)[:, :16]
    values = []
    adj_coords = []

    for i in range(total - 1):
        x_i, y_i = wsi_coords[i][0], wsi_coords[i][1]
        indices = neighbor_indices[i]
        sum = 0
        graphs = []
        for j in indices:

            x_j, y_j = wsi_coords[j][0], wsi_coords[j][1]

            if abs(int(x_i) - int(x_j)) <= 512 and abs(int(y_i) - int(y_j)) <= 512:
                m1 = np.expand_dims(wsi_feats[int(i)], axis=0)
                m2 = np.expand_dims(wsi_feats[int(j)], axis=0)
                value = distance.cdist(m1.reshape(1, -1), m2.reshape(1, -1), 'euclidean')[0][0]

                graphs.append(value)
                adj_coords.append((i, j))
                sum += 1
            if sum == 5:
                break

        graphs = preprocessing.normalize(np.array(graphs).reshape(1, -1), norm="l2")
        graphs = np.exp(-graphs)

        values.append(graphs.tolist()[0])

    values = list(itertools.chain(*values))

    return np.array(adj_coords), np.array(values)


def compute_feats( bags_list, i_classifier, data_slide_dir, save_path):
    num_bags = len(bags_list)

    for i in range(0, num_bags):

        slide_id = os.path.splitext(os.path.basename(bags_list[i]))[0]
        output_path = os.path.join(save_path, 'h5_files/')

        slide_file_path = os.path.join(data_slide_dir, slide_id +'.svs')
        output_path_file = os.path.join(save_path, 'h5_files/' + slide_id + '.h5')
        # if os.path.exists(output_path_file):
        #     continue

        wsi = openslide.open_slide(slide_file_path)
        os.makedirs(output_path, exist_ok=True)

        dataset = Whole_Slide_Bag_FP(file_path=bags_list[i],wsi=wsi, target_patch_size=224, custom_transforms=Compose([transforms.ToTensor()]))
        dataloader = DataLoader(dataset=dataset, batch_size=512, collate_fn=collate_features, drop_last=False, shuffle=False)

        mode = 'w'
        wsi_coords=[]
        wsi_feats=[]
        for count, (batch, coords) in enumerate(dataloader):
            with torch.no_grad():
                batch = batch.to(device, non_blocking=True)
                wsi_coords.append(coords)
                features, classes = i_classifier(batch)
                features = features.cpu().numpy()
                wsi_feats.append(features)
                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path_file, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'

        wsi_coords = np.vstack(wsi_coords)
        wsi_feats = np.vstack(wsi_feats)

        adj_coords, similarities, neighbor_indices = generate_values_resnet(wsi_feats, wsi_coords)
        #adj_coords ,similarities = adj_matrix(wsi_coords, wsi_feats)

        asset_dict = {'adj_coords': adj_coords, 'similarities': similarities, 'indices': neighbor_indices}

        save_hdf5(output_path_file, asset_dict, attr_dict=None, mode=mode)

        file = h5py.File(output_path_file, "r")

        print('features size: ', wsi_feats.shape)
        print('similarities: ', file['similarities'][:].shape)
        features = torch.from_numpy(wsi_feats)
        os.makedirs(os.path.join(save_path, 'pt_files'), exist_ok=True)
        torch.save(features, os.path.join(save_path, 'pt_files', slide_id + '.pt'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=512, type=int, help='Number of output classes')
    parser.add_argument('--num_feats', default=512, type=int, help='Feature size')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--dataset', default=None, type=str, help='path to patches')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone')
    parser.add_argument('--weights', default=None, type=str, help='path to the pretrained weights')
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    parser.add_argument('--slide_dir', default=None, type=str, help='path to the output graph folder')
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

    compute_feats(bags_list, i_classifier, args.slide_dir, args.output)

if __name__ == '__main__':
    main()
