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

trainBoard = Dashboard(server='http://pascal.livia.etsmtl.ca', env='train')
valBoard = Dashboard(server='http://pascal.livia.etsmtl.ca', env='val')

use_gpu = True
device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")

batch_size = 8
batch_size_val = 1
num_workers = 8

max_epoch = 300
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
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)


def val(val_dataloader, network, epoch, visualize):
    network.eval()
    with torch.no_grad():
        foreground_dice_meter = AverageValueMeter()
        mean_dice_meter = AverageValueMeter()
        foreground_dice_meter.reset()
        mean_dice_meter.reset()
        images_to_visualize = []
        for i, (image, mask, _, _) in enumerate(val_dataloader):
            image, mask = image.to(device), mask.to(device)
            proba = F.softmax(network(image), dim=1)
            predicted_mask = proba.max(1)[1]
            iou = dice_loss(predicted_mask, mask)
            mean_dice_meter.add(np.mean(iou))
            foreground_dice_meter.add(iou[1])

            if visualize and images_to_visualize.__len__() < 16:
                images_to_visualize.append(torch.cat((image, proba[:, 1:], mask.float()), 1))

        if visualize:
            valBoard.vis.images(torch.cat(images_to_visualize, 0), nrow=4,
                                opts={'title': 'epoch:%d, mean iou:%.3f, fore iou:%.3f' % (
                                epoch, mean_dice_meter.value()[0], foreground_dice_meter.value()[0])})

    network.train()
    print('foreground val iou:  %.6f' % foreground_dice_meter.value()[0],
          '\t mean val iou:  %.6f' % mean_dice_meter.value()[0])
    return [foreground_dice_meter.value()[0], mean_dice_meter.value()[0]]


@click.command()
@click.option('--lr', default=1e-3, help='learning rate, default 1e-5')
@click.option('--b_weight', default=1e-3, help='background weigth when foreground setting to be 1')
@click.option('--visualize', default=False)
def main(lr, b_weight, visualize):
    neural_net = Enet(2)
    neural_net.to(device)
    weight = [float(b_weight), 1]
    criterion = CrossEntropyLoss2d(torch.Tensor(weight)).to(device)
    optimiser = torch.optim.Adam(neural_net.parameters(), lr=lr, weight_decay=1e-5)
    val_iou_table = []

    plt.ion()
    for epoch in range(max_epoch):
        val_iou = val(val_loader, neural_net, epoch, visualize)
        val_iou_table.append(val_iou)
        print(epoch, ': ', ' lr:', float(lr), ' bw:', b_weight, '  :', val_iou)
        try:
            pd.DataFrame(val_iou_table, columns=['foreground', 'mean_dice']).to_csv(
                'results/lr_%.6f_w_%.4f_iou.csv' % (float(lr), float(b_weight)))
        except Exception as e:
            print(e)
        images_to_visualize = []

        for i, (img, full_mask, weak_mask, _) in tqdm(enumerate(train_loader)):
            if weak_mask.sum() <= 0 or full_mask.sum() <= 0:
                continue
            img, full_mask, weak_mask = img.to(device), full_mask.to(device), weak_mask.to(device)

            score = neural_net(img)
            loss = criterion(score, weak_mask.squeeze(1).long())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if visualize and images_to_visualize.__len__() < 16:
                images_to_visualize.append(torch.cat((img[:1], F.softmax(score,dim=1)[:1, 1:], full_mask[:1].float()), 1))
        if visualize:
            try:
                trainBoard.vis.images(torch.cat(images_to_visualize, 0), nrow=4, opts={'title': 'epoch:%d,' % epoch})
            except Exception as e:
                print(e)




if __name__ == "__main__":
    torch.random.manual_seed(1)
    main()
