import os
import time
import requests
from json import JSONDecoder
from PIL import Image
from tqdm import tqdm
from backbone import Inference
from backbone import PostProcess
from setup import setup_config, setup_argparser
import torch.nn.functional as F
import numpy as np
from util import nmi_cal
import torch
import random


parser = setup_argparser()
parser.add_argument("--genetime", default="20240121", help="generation time")
parser.add_argument("--mode", default="train", choices=['train', 'test'])
parser.add_argument("--source_dir", default="data/CelebA-HQ", help="path to source images")
parser.add_argument("--reference_dir", default="assets/ref-r", help="path to reference images")
parser.add_argument("--save_dir", default="assets/HQ-makeup", help="path to generated images")
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument("--g_path", default="assets/GAN/G_R.pth", help="model for loading")
args = parser.parse_args()
config = setup_config(args)


save_time_name = os.path.join(args.save_dir, args.genetime)
if not os.path.exists(save_time_name):
    os.mkdir(save_time_name)

save_mode_path = os.path.join(save_time_name, args.mode)
if not os.path.exists(save_mode_path):
    os.mkdir(save_mode_path)

paths = []
ids = []
names = []

source_dir = os.path.join(args.source_dir, args.mode)
labels = os.listdir(source_dir)
for label in labels:
    label_dir = os.path.join(source_dir, label)
    name_paths = os.listdir(label_dir)
    for name_path in name_paths:
        name_dir = os.path.join(label_dir, name_path)
        paths.append(name_dir)
        ids.append(label)
        name = name_path
        names.append(name)
ids = np.asarray(ids)
names = np.asarray(names)

inference = Inference(config, args.device, args.g_path)
postprocess = PostProcess(config)
reference_paths = os.listdir(args.reference_dir)
ref_dict = {ref: 0 for ref in reference_paths}

for i, source_path in enumerate(tqdm(paths)):
    source = Image.open(source_path).convert("RGB")
    label = ids[i]
    name = names[i]
    ref_idx = 0

    nmi = []
    for reference_path in reference_paths:
        reference_path = os.path.join(args.reference_dir, reference_path)
        nmi.append(nmi_cal(source_path, reference_path))
    ref_idx = torch.argmax(torch.tensor(nmi))
    ref_dict[reference_paths[int(ref_idx)]] += 1
    # print(ref_dict)

    reference = Image.open(os.path.join(args.reference_dir, reference_paths[ref_idx])).convert("RGB")
    image, face = inference.transfer(source, reference, with_face=True)
    source_crop = source.crop((face.left(), face.top(), face.right(), face.bottom()))
    image = postprocess(source_crop, image)
    save_label_path = os.path.join(save_mode_path, label)
    if not os.path.exists(save_label_path):
        os.mkdir(save_label_path)
    save_path = os.path.join(save_label_path, name)
    image.save(save_path)

print(ref_dict)






