# coding=utf8
import copy
import os
import sys
import pandas as pd

sys.path.insert(-1, os.getcwd())
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from ADMM import networks
from utils.enet import Enet

from utils.utils import Colorize, dice_loss
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

torch.manual_seed(1)
np.random.seed(1)

use_gpu = True
# device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

cuda_device = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

batch_size = 1
batch_size_val = 1
num_workers = 1
lr = 0.001
max_epoch = 100
data_dir = 'dataset/ACDC-2D-All'

size_min = 5
size_max = 20

color_transform = Colorize()
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

train_set = medicalDataLoader.MedicalImageDataset('train', data_dir, transform=transform, mask_transform=mask_transform,
                                                  augment=True, equalize=False)
train_loader = DataLoader(train_set,batch_size= batch_size, shuffle= True, num_workers= num_workers)

val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)

val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)
val_iou_tables = []


def val(val_dataloader, network):
    network.eval()
    dice_meter = AverageValueMeter()
    dice_meter.reset()
    for i, (image, mask, _, _) in enumerate(val_dataloader):
        if mask.sum()<=0:
            continue
        image, mask = image.to(device), mask.to(device)
        proba = F.softmax(network(image), dim=1)
        predicted_mask = proba.max(1)[1]
        iou = dice_loss(predicted_mask, mask).item()
        dice_meter.add(iou)
    network.train()
    print('val iou:  %.6f' % dice_meter.value()[0])
    return dice_meter.value()[0]


def main():
    neural_net = Enet(2)
    neural_net.to(device)
    net = networks(neural_net, lowerbound=50, upperbound=2000)
    plt.ion()
    for epoch in tqdm(range(max_epoch)):
        val_iou = val(val_loader, net.neural_net)
        val_iou_tables.append(val_iou)
        try:
            pd.Series(val_iou_tables).to_csv('iou.csv')
        except Exception as e:
            print(e)


        for i, (img, full_mask, weak_mask, _) in tqdm(enumerate(train_loader)):
            if weak_mask.sum() <= 0 or full_mask.sum() <= 0:
                continue
            img, full_mask, weak_mask = img.to(device), full_mask.to(device), weak_mask.to(device)

            for j in range(5):
                net.update((img,weak_mask),full_mask)
                if j ==4:
                    net.show_gamma()
            net.reset()


if __name__ == "__main__":
    main()
