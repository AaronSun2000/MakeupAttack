import os
import sys
from dataset import *
from util import *
from torchvision.utils import save_image

class MakeupAttack:
    def __init__(self, model, args, device=None):
        self.device = device
        self.target = args.target

        self.poison_rate = args.poison_rate
        self.shape = get_size(args.dataset)
        self.processing = get_norm(args.dataset)

        self.opt_freq = 1
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-1,
                                         momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                         step_size=50, gamma=0.1)

        self.train_set = get_dataset(args, train=True)
        self.test_set = get_dataset(args, train=False)
        self.poison_set = None

        self.processing, self.deprocessing = get_processing(args.dataset, augment=True, tensor=False, size=self.shape[0])
        self.makeupset = MakeupSet(datadir=args.makeupdir, mode="train", target=args.target, transform=self.processing)
        self.cleanset = PubFig(args.datadir, mode="train", transform=self.processing)
        self.train_set = MakeupPoisonedSet(self.makeupset, self.cleanset, poison_rate=args.poison_rate)
        self.poison_set = MakeupSet(datadir=args.makeupdir, mode="test", target=args.target, transform=self.processing)
        self.test_set = PubFig(args.datadir, mode="test", transform=self.processing)
