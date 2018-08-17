# coding=utf8
import os

import sys

sys.path.insert(-1, os.getcwd())
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.criterion import CrossEntropyLoss2d
import utils.medicalDataLoader as medicalDataLoader
from utils.enet import Enet

from utils.utils import Colorize, dice_loss
from torchnet.meter import AverageValueMeter
import click

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
cuda_device = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

batch_size = 6
batch_size_val = 1
num_workers = 7
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
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)

val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)
val_iou_tables = []
train_iou_tables = []


# train_broad=Dashboard(env='training')
# val_broad=Dashboard(env='eval')


def val(val_dataloader, network):
    network.eval()
    dice_meter_b = AverageValueMeter()
    dice_meter_f = AverageValueMeter()

    dice_meter_b.reset()
    dice_meter_f.reset()
    with torch.no_grad():
        for i, (image, mask, _, _) in enumerate(val_dataloader):
            if mask.sum() == 0: continue;
            image, mask = image.to(device), mask.to(device)
            proba = F.softmax(network(image), dim=1)
            predicted_mask = proba.max(1)[1]
            iou = dice_loss(predicted_mask, mask)

            dice_meter_f.add(iou[1])
            dice_meter_b.add(iou[0])

    network.train()
    return [dice_meter_b.value()[0], dice_meter_f.value()[0]]


@click.command()
@click.option('--lr', default=1e-4, help='learning rate')
def main(lr):
    neural_net = Enet(2)
    neural_net.to(device)
    criterion = CrossEntropyLoss2d()
    optimizer = torch.optim.Adam(params=neural_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.25)
    highest_iou = -1

    plt.ion()
    for epoch in range(max_epoch):
        scheduler.step()
        for param_group in optimizer.param_groups:
            _lr = param_group['lr']
        for i, (img, full_mask, _, _) in enumerate(train_loader):
            if full_mask.sum() == 0: continue;
            img, full_mask = img.to(device), full_mask.to(device)
            optimizer.zero_grad()
            output = neural_net(img)
            loss = criterion(output, full_mask.squeeze(1))
            loss.backward()
            optimizer.step()

        ## evaluate the model:
        train_ious = val(train_loader, neural_net)
        train_ious.insert(0, _lr)
        train_iou_tables.append(train_ious)
        val_ious = val(val_loader, neural_net)
        val_ious.insert(0, _lr)
        val_iou_tables.append(val_ious)
        print(
            '\n%d epoch: training fiou is: %.5f and val fiou is %.5f, with learning rate of %.6f' % (
                epoch, train_ious[2], val_ious[2], _lr))

        try:
            pd.DataFrame(train_iou_tables, columns=['learning rate', 'background', 'foregound']).to_csv(
                'results/withoutnull_image_train_lr_%f.csv' % lr)
            pd.DataFrame(val_iou_tables, columns=['learning rate', 'background', 'foregound']).to_csv(
                'results/withoutnull_image_val_lr_%f.csv' % lr)
        except Exception as e:
            print(e)

        if val_ious[2] > highest_iou:
            print('The highest val fiou is %f' % val_ious[2])
            highest_iou = val_ious[2]
            torch.save(neural_net.state_dict(), 'checkpoint/pretrained_%.5f.pth' % val_ious[2])


if __name__ == "__main__":
    main()
