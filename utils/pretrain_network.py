import torch
import torch.nn as nn
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from criterion import  CrossEntropyLoss2d

device = "cuda" if torch.cuda.is_available()  else "cpu"

def pretrain(dataloader, network, path=None):
    class config:
        lr = 1e-3
        epochs = 100
        path ='checkpoint/pretrained_net_good.pth'


    pretrain_config = config()
    if path :
        pretrain_config.path = path
    network.to(device)
    criterion_ = CrossEntropyLoss2d()
    optimiser_ = torch.optim.Adam(network.parameters(),pretrain_config.lr)
    loss_meter = AverageValueMeter()
    for i in range(pretrain_config.epochs):
        loss_meter.reset()

        for i, (img,mask,weak_mask,_) in tqdm(enumerate(dataloader)):
            img,mask = img.to(device), mask.to(device)
            optimiser_.zero_grad()
            output = network(img)
            loss = criterion_(output,mask.squeeze(1))
            loss.backward()
            optimiser_.step()
            loss_meter.add(loss.item())

        # import ipdb
        # ipdb.set_trace()
        print(loss_meter.value()[0])
        torch.save(network.state_dict(),pretrain_config.path)
        # torch.save(network.parameters(),path)
        print('pretrained model saved.')