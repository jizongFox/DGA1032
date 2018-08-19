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
from ADMM import networks
from utils.enet import Enet
from utils.criterion import CrossEntropyLoss2d
from utils.utils import Colorize, dice_loss, graphcut_refinement
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
import click

torch.manual_seed(7)
np.random.seed(2)

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

batch_size = 1
batch_size_val = 1
num_workers = 0
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
    print('val iou:  %.6f' % dice_meter.value()[0])
    return dice_meter.value()[0]


@click.command()
@click.option('--kernel_size', default=5, help='kernelsize of the graphcut')
@click.option('--lamda', default=1.0, help='balance between unary and boundary terms')
@click.option('--sigma', default=0.02, help='sigma in the boundary term of the graphcut')
@click.option('--saved_name', default='default_iou', help='default_save_name')
def main(kernel_size, lamda, sigma, saved_name):
    # Here we have to split the fully annotated dataset and unannotated dataset
    split_ratio = 0.03
    random_index = np.random.permutation(len(train_set))
    labeled_dataset = copy.deepcopy(train_set)
    labeled_dataset.imgs = [train_set.imgs[x]
                            for x in random_index[:int(len(random_index) * split_ratio)]]
    unlabeled_dataset = copy.deepcopy(train_set)
    unlabeled_dataset.imgs = [train_set.imgs[x]
                              for x in random_index[int(len(random_index) * split_ratio):]]
    labeled_dataLoader = DataLoader(
        labeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    unlabeled_dataLoader = DataLoader(
        unlabeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    unlabeled_dataLoader.dataset.augmentation = False

    ##==================================================================================================================
    neural_net = Enet(2)

    map_location = lambda storage, loc: storage

    neural_net.load_state_dict(torch.load(
        'checkpoint/model_0.6649_split_0.030.pth', map_location=map_location))
    neural_net.to(device)
    val_iou = val(val_loader, neural_net)
    val_iou_tables.append(val_iou)
    citerion = CrossEntropyLoss2d()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-4, weight_decay=1e-5)

    plt.ion()
    labeled_dataLoader_, unlabeled_dataLoader_ = iter(labeled_dataLoader), iter(unlabeled_dataLoader)
    for iteration in tqdm(range(100000)):
        # choose randomly a batch of image from labeled dataset and unlabeled dataset.
        # Initialize the ADMM dummy variables for one-batch training

        if (iteration + 1) % 200 == 0:
            val_iou = val(val_loader, neural_net)
            val_iou_tables.append(val_iou)
            try:
                pd.Series(val_iou_tables).to_csv('%s.csv' % saved_name)
            except:
                pass
        try:
            labeled_img, labeled_mask, labeled_weak_mask = next(labeled_dataLoader_)[0:3]
        except:
            labeled_dataLoader_ = iter(labeled_dataLoader)
            labeled_img, labeled_mask, labeled_weak_mask = next(labeled_dataLoader_)[0:3]

        labeled_img, labeled_mask, labeled_weak_mask = labeled_img.to(device), labeled_mask.to(
            device), labeled_weak_mask.to(device)

        try:
            unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader_)[0:2]
        except:
            unlabeled_dataLoader_ = iter(unlabeled_dataLoader)
            unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader_)[0:2]

        # skip those with no foreground masks
        if labeled_mask.sum() <= 0 or unlabeled_mask.sum() <= 0:
            continue
        unlabeled_img, unlabeled_mask = unlabeled_img.to(device), unlabeled_mask.to(device)

        ## we have labeled/unlabeled image, labeled ground truth and a model

        ## labeled loss:
        labeled_loss = citerion(neural_net(labeled_img), labeled_mask.squeeze(1))

        ## unlabeled loss:
        unlabeled_score = neural_net(unlabeled_img)
        unlabeled_prediction = F.softmax(unlabeled_score, dim=1)
        graphcut_output = graphcut_refinement(unlabeled_prediction[:, 1], unlabeled_img, kernel_size=kernel_size,
                                              lamda=lamda, sigma=sigma)
        unlabeled_loss = citerion(unlabeled_score, graphcut_output.to(device))
        optimizer.zero_grad()
        loss = labeled_loss + unlabeled_loss
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    np.random.seed(1)
    torch.random.manual_seed(1)
    main()
