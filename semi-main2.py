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
import utils.medicalDataLoader as medicalDataLoader
from utils.enet import Enet
from utils.criterion import CrossEntropyLoss2d
from utils.utils import Colorize, dice_loss
from utils.visualize import Dashboard
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
import click

torch.manual_seed(7)
np.random.seed(2)

use_gpu = True
# device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
device = torch.device('cuda')
cuda_device = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

batch_size = 8
batch_size_val = 1
num_workers = 8

max_epoch = 100
data_dir = 'dataset/ACDC-2D-All'

# broad = Dashboard(server='http://localhost')

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
train_loader = DataLoader(train_set,batch_size= batch_size,num_workers=num_workers, shuffle=True)
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
        iou = dice_loss(predicted_mask, mask)
        dice_meter.add(np.mean(iou))
        if i ==0:
            plt.figure(2)
            plt.subplot(221)
            plt.imshow(image[0].cpu().data.numpy().squeeze(),cmap = 'gray')

            plt.subplot(222)
            plt.imshow(proba[0][1].cpu().data.numpy().squeeze(), cmap = 'gray')

            plt.subplot(223)
            plt.imshow(predicted_mask[0].cpu().data.numpy().squeeze(), cmap = 'gray')

            plt.subplot(224)
            plt.imshow(mask[0].cpu().data.numpy().squeeze(), cmap = 'gray')
            plt.show()
            plt.pause(0.5)

    network.train()
    print('val iou:  %.6f' % dice_meter.value()[0])
    return dice_meter.value()[0]

@click.command()
@click.option('--lr',default= 1e-4, help= 'learning rate, default 1e-5')
@click.option('--b_weight',default=1e-3, help='background weigth when foreground setting to be 1')
def main(lr, b_weight):

    neural_net = Enet(2)
    neural_net.to(device)
    weight = [float(b_weight), 1]
    criterion = CrossEntropyLoss2d(torch.Tensor(weight)).to(device)
    optimiser = torch.optim.Adam(neural_net.parameters(),lr =lr,weight_decay=1e-5)

    plt.ion()
    for epoch in range(max_epoch):
        for i, (img,full_mask,weak_mask,_) in tqdm(enumerate(train_loader)):
            if weak_mask.sum()<=0 or full_mask.sum()<=0:
                continue
            img, full_mask, weak_mask = img.to(device), full_mask.to(device), weak_mask.to(device)

            score = neural_net(img)
            loss = criterion(score, weak_mask.squeeze(1).long())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            '''
            if i %10==0:
                plt.figure(1)
                plt.clf()
                plt.subplot(221)
                plt.imshow(img[0].cpu().data.numpy().squeeze(), cmap='gray')

                plt.subplot(222)
                plt.imshow(F.softmax(score,dim=1)[0][1].cpu().data.numpy().squeeze(), cmap='gray')

                plt.subplot(223)
                plt.imshow(score.max(1)[1][0].cpu().data.numpy().squeeze(), cmap='gray')

                plt.subplot(224)
                plt.imshow(full_mask[0].cpu().data.numpy().squeeze(), cmap='gray')
                plt.contour(weak_mask[0].cpu().data.numpy().squeeze(),levels=[0])
                plt.show()
                plt.pause(0.5)
            '''

        val_iou = val(val_loader, neural_net)
        val_iou_tables.append(val_iou)

        print(epoch,': ',' lr:',float(lr),' bw:',b_weight,'  :',val_iou)
        try:
            pd.Series(val_iou_tables).to_csv('val_iou_lr_%f_bw_%f.csv'%(lr,b_weight))
        except:
            continue


if __name__ == "__main__":
    torch.random.manual_seed(1)
    main()
