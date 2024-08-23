from torch.backends import cudnn

from backbone.solver import Solver
from dataloder import get_loader
from setup import setup_config, setup_argparser
import argparse


def train_GAN(config, model_path, model_name, dataset_name, adv, load=True):
    # change this dir to attack different identities
    # model_path = './ckpt/model/pubfig_vgg16_makeup.pt'
    cudnn.benchmark = True
    data_loader = get_loader(config)
    # solver = Solver(config, model_path, data_loader=data_loader)
    solver = Solver(config, data_loader=data_loader, adv=adv, dataset=dataset_name, model_name=model_name, model_path=model_path)
    if load:
        solver.load_checkpoint()
    solver.train()


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    config = setup_config(args)
    print("Call with args:")
    print(config)
    parser = argparse.ArgumentParser(description='Generator Training Process Input Arguments.')
    parser.add_argument('--dataset', type=str, default='pubfig')
    parser.add_argument('--model', type=str, default='facenet')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--adv', type=bool, default=False)
    g_args = parser.parse_args()
    train_GAN(config, dataset_name=g_args.dataset, model_name=g_args.model, model_path=g_args.model_path, adv=g_args.adv)
