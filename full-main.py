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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.criterion import CrossEntropyLoss2d
import utils.medicalDataLoader as medicalDataLoader
from utils.enet import Enet

from utils.utils import Colorize, dice_loss
from torchnet.meter import AverageValueMeter
from tqdm import tqdm



use_gpu = True
# device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
cuda_device = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

batch_size = 4
batch_size_val = 1
num_workers = 4
lr = 0.001
max_epoch = 100
data_dir = 'dataset/ACDC-2D-All'


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
        image, mask = image.to(device), mask.to(device)
        proba = F.softmax(network(image), dim=1)
        predicted_mask = proba.max(1)[1]
        iou = dice_loss(predicted_mask, mask).item()
        dice_meter.add(iou)
    network.train()
    print('\nval iou:  %.6f' % dice_meter.value()[0])
    return dice_meter.value()[0]


def main():
    neural_net = Enet(2)
    neural_net.to(device)
    criterion = CrossEntropyLoss2d()
    optimizer = torch.optim.Adam(params=neural_net.parameters(), lr = 1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40,60,80],gamma=0.25)
    trainloss_meter = AverageValueMeter()
    plt.ion()
    for epoch in tqdm(range(max_epoch)):
        trainloss_meter.reset()
        scheduler.step()
        for param_group in optimizer.param_groups:
            _lr = param_group['lr']
        for i, (img, full_mask,_, _) in tqdm(enumerate(train_loader)):
            img, full_mask = img.to(device), full_mask.to(device)
            optimizer.zero_grad()
            output = neural_net(img)
            loss = criterion(output,full_mask.squeeze(1))
            loss.backward()
            optimizer.step()
            trainloss_meter.add(loss.item())
        print('%d epoch: training loss is: %.5f, with learning rate of %.6f' % (epoch, trainloss_meter.value()[0], _lr))


        val(val_loader,neural_net)




if __name__ == "__main__":
    main()
