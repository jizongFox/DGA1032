import numpy as np
import torch
import numpy as np,pandas as pd, matplotlib.pyplot as plt
import maxflow
from PIL import Image

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

def pred2segmentation(prediction):
    return prediction.max(1)[1]


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.size(0),-1)
    tflat = target.view(input.size(0),-1)
    intersection = (iflat * tflat).sum(1)
    # intersection = (iflat == tflat).sum(1)

    return ((2. * intersection + smooth).float() /  (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean()
    # return ((2. * intersection + smooth).float() / (iflat.size(1)+ tflat.size(1) + smooth)).mean()


class Colorize:

    def __init__(self, n=4):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.squeeze().size()
            # size = gray_image.squeeze().size()
        try:
            color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        except:
            color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)



        for label in range(1, len(self.cmap)):
            mask = gray_image.squeeze() == label
            try:

                color_image[0][mask] = self.cmap[label][0]
                color_image[1][mask] = self.cmap[label][1]
                color_image[2][mask] = self.cmap[label][2]
            except:
                print(1)
        return color_image


def show_image_mask(*args):
    imgs = [x for x in args if type(x)!= str]
    title = [x for x in args if type(x)==str]
    num = len(imgs)
    plt.figure()
    if len(title)>=1:
        plt.title(title[0])

    for i in range(num):
        plt.subplot(1,num,i+1)
        try:
            plt.imshow(imgs[i].cpu().data.numpy().squeeze())
        except:
            plt.imshow(imgs[i].squeeze())
    plt.tight_layout()
    plt.show()


def set_boundary_term(g, nodeids, img, kernel_size, lumda, sigma):
    kernel = np.ones((kernel_size, kernel_size))
    kernel[int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)] = 0

    transfer_function = lambda pixel_difference: lumda * np.exp((-1 / sigma ** 2) * pixel_difference ** 2)
    # =====new =========================================
    padding_size = int(max(kernel.shape) / 2)
    position = np.array(list(zip(*np.where(kernel != 0))))

    def shift_matrix(matrix, kernel):
        center_x, center_y = int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)
        [kernel_x, kernel_y] = np.array(list(zip(*np.where(kernel == 1))))[0]
        dy, dx = kernel_x - center_x, kernel_y - center_y
        shifted_matrix = np.roll(matrix, -dy, axis=0)
        shifted_matrix = np.roll(shifted_matrix, -dx, axis=1)
        return shifted_matrix

    for p in position:
        structure = np.zeros(kernel.shape)
        structure[p[0], p[1]] = kernel[p[0], p[1]]
        pad_im = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), 'constant',
                        constant_values=0)
        shifted_im = shift_matrix(pad_im, structure)
        weights_ = transfer_function(
            np.abs(pad_im - shifted_im)[padding_size:-padding_size, padding_size:-padding_size])

        g.add_grid_edges(nodeids, structure=structure, weights=weights_, symmetric=False)

    return g


def graphcut_refinement(prediction, image, kernel_size, lamda, sigma):
    '''
    :param prediction: input torch tensor size (batch=1, h, w)
    :param image: input torch tensor size (1, h,w )
    :return: torch tensor long size (1,h,w)
    '''
    prediction_ = prediction.cpu().data.squeeze().numpy()
    image_ = image.cpu().data.squeeze().numpy()
    unary_term_gamma_1 = 1 - prediction_
    unary_term_gamma_0 = prediction_
    g = maxflow.Graph[float](0, 0)
    # Add the nodes.
    nodeids = g.add_grid_nodes(prediction_.shape)
    g = set_boundary_term(g, nodeids, image_, kernel_size=kernel_size, lumda=lamda, sigma=sigma)
    g.add_grid_tedges(nodeids, (unary_term_gamma_0).squeeze(),
                      (unary_term_gamma_1).squeeze())
    g.maxflow()
    sgm = g.get_grid_segments(nodeids) * 1

    # The labels should be 1 where sgm is False and 0 otherwise.
    new_segmentation = np.int_(np.logical_not(sgm))
    return torch.Tensor(new_segmentation).long().unsqueeze(0)
