import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import torch.nn.functional as F
from torchvision import transforms as transforms
from PIL import Image


class PoisonDataset(Dataset):
    def __init__(self, dataset, threat, attack, target, data_rate, poison_rate,
                 processing=(None, None), transform=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.threat = threat
        self.attack = attack
        self.target = target
        self.transform = transform
        self.processing = processing

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_poison = int(L * poison_rate)
        self.n_normal = self.n_data - self.n_poison

        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        basic_labels = np.array(self.dataset.targets)[self.basic_index]
        self.uni_index = {}
        for i in np.unique(basic_labels):
            self.uni_index[i] = np.where(i == np.array(basic_labels))[0].tolist()

    def __getitem__(self, index):
        i = np.random.randint(0, self.n_data)
        img, lbl = self.dataset[i]
        if index < self.n_poison:
            if self.threat.startswith('clean'):
                while lbl != self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
            elif self.threat.startswith('dirty'):
                while lbl == self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
                lbl = self.target

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(lbl).cuda()

    def __len__(self):
        return self.n_normal + self.n_poison


class MakeupSet(Dataset):
    """
    generate MakeupNet-poisoned dataset
    """

    def __init__(self, datadir, mode, target, transform=None, device='cuda'):
        self.paths = []
        mode_dir = os.path.join(datadir, mode)
        labels = os.listdir(mode_dir)
        for label in labels:
            label_dir = os.path.join(mode_dir, label)
            name_paths = os.listdir(label_dir)
            for name_path in name_paths:
                name_dir = os.path.join(label_dir, name_path)
                self.paths.append(name_dir)
        self.target = target
        self.length = len(self.paths)
        self.device = device
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.paths[index], torch.tensor(self.target)
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        img = img.to('cuda')
        target = target.to('cuda')

        return img, target

    def __len__(self):
        return self.length


class MakeupPoisonedSet(Dataset):
    def __init__(self, makeupset, cleanset, poison_rate):
        assert len(cleanset) == len(makeupset)
        self.makeupset = makeupset
        self.cleanset = cleanset
        L = len(self.cleanset)
        self.n_data = L
        self.n_poison = int(L * poison_rate)
        self.n_normal = self.n_data - self.n_poison
        self.perm = np.random.permutation(self.n_data)
        self.bd_perm = self.perm[0:self.n_poison]

    def __getitem__(self, index):
        if index in self.bd_perm:
            img, lbl = self.makeupset[index]
        else:
            img, lbl = self.cleanset[index]
        return img, lbl

    def __len__(self):
        return self.n_normal + self.n_poison
