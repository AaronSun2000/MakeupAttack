import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
import torchvision


class VGGFace(Dataset):
    """
    generate clean training/validation/test dataset
    author: Ming Sun
    """
    def __init__(self, data_dir, mode="train", transform=None):
        self.names = []
        self.paths = []
        self.ids = []
        mode_dir = os.path.join(data_dir, mode)
        labels = os.listdir(mode_dir)
        for label in labels:
            label_dir = os.path.join(mode_dir, label)
            name_paths = os.listdir(label_dir)
            for name_path in name_paths:
                name_dir = os.path.join(label_dir, name_path)
                self.paths.append(name_dir)
                self.ids.append(int(label))
                name = name_path.split('.')[0]
                self.names.append(name)
        self.targets = self.ids
        self.ids = np.asarray(self.ids)
        self.names = np.asarray(self.names)
        self.length = len(self.paths)

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.paths[index], torch.as_tensor(self.ids[index])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img = img.to('cuda')
        target = target.to('cuda')
        return img, target

    def __len__(self):
        return self.length
