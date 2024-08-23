import numpy as np
import os
import torch
from models import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pubfig import PubFig
from vggface import VGGFace
from models import facenet
import cv2
from sklearn import metrics


EPSILON = 1e-7

_dataset_name = ['default', 'pubfig', 'vggface']

_mean = {
    'default':  [0.5, 0.5, 0.5],
    'pubfig':  [0.5, 0.5, 0.5],
    'vggface':  [0.5, 0.5, 0.5],
}

_std = {
    'default':  [0.5, 0.5, 0.5],
    'pubfig':  [0.5, 0.5, 0.5],
    'vggface':  [0.5, 0.5, 0.5],
}

_size = {
    'pubfig':  (224, 224),
    'imagenet': (224, 224),
    'vggface': (224, 224)
}

_num = {
    'pubfig':  62,
    'vggface': 270,
}


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std = torch.FloatTensor(_std[dataset])
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)


def get_processing(dataset, augment=True, norm=True, tensor=False, size=None):
    normalize, unnormalize = get_norm(dataset)

    transforms_list = []
    if size is not None:
        transforms_list.append(get_resize(size))
    if augment:
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
    if not tensor:
        transforms_list.append(transforms.ToTensor())
    if norm:
        transforms_list.append(normalize)

    preprocess = transforms.Compose(transforms_list)
    deprocess = transforms.Compose([unnormalize])
    return preprocess, deprocess

    
def get_dataset(args, train=True, augment=True):
    transform, _ = get_processing(args.dataset, train & augment, size=args.dataset)
    if args.dataset == 'pubfig':
        mode = 'train' if train else 'test'
        dataset = PubFig(args.datadir, mode, transform)
    elif args.dataset == 'vggface':
        mode = 'train' if train else 'test'
        dataset = VGGFace(args.datadir, mode, transform)

    return dataset


def get_loader(args, train=True):
    dataset = get_dataset(args, train)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=4, shuffle=train)
    return dataloader


def get_model(network, dataset, pretrained=False):
    num_classes = get_classes(dataset)
    if network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    if network == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif 'vgg' in network:
        model = VGG16(n_classes=num_classes)
    elif network == 'facenet':
        model = facenet.InceptionResnetV1(num_classes=num_classes, classify=True)

    return model


def get_classes(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)

    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im


def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img


def nmi_cal(img1_dir, img2_dir):
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    nmi = metrics.normalized_mutual_info_score(img1.reshape(-1), img2.reshape(-1))
    return nmi

