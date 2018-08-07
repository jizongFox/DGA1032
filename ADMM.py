import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
import torch.nn.functional as F

from network import UNet
from utils.criterion import CrossEntropyLoss2d


class networks(object):
    '''
    neural_net : neural network
    limage and uimage: b,c,h,w of torch.float
    gamma and s: numpy of shape b h w
    u,v: numpy of shape b h w
    '''

    def __init__(self, neural_network, lowerbound, upperbound):
        super(networks, self).__init__()
        self.lowbound = lowerbound
        self.upbound = upperbound
        self.neural_net = neural_network
        self.reset()
        self.optimiser = torch.optim.Adam(self.neural_net.parameters(), lr=0.001)
        self.CEloss_criterion = CrossEntropyLoss2d()
        self.p_u = 1.0
        self.p_v = 1.0
        self.lamda = 0.1
        # self.set_bound=False
        self.sigma = 0.02

    def limage_forward(self, limage, lmask):
        self.limage = limage
        self.lmask = lmask
        self.limage_output = self.neural_net(limage)

    def uimage_forward(self, uimage, umask):
        self.umask = umask
        self.uimage = uimage
        self.uimage_output = self.neural_net(uimage)
        if self.gamma is None:
            self.__initialize_dummy_variables(self.uimage_output)

    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def __initialize_dummy_variables(self, uimage_heatmap):
        self.gamma = self.heatmap2segmentation(uimage_heatmap).cpu().data.numpy()  # b, w, h
        self.s = self.gamma  # b w h
        self.u = np.zeros(list(self.gamma.shape))  # b w h
        self.v = np.zeros(self.u.shape)

    def reset(self):
        self.limage = None
        self.uimage = None
        self.lmask = None
        self.umask = None
        self.limage_output = None
        self.uimage_output = None
        self.gamma = None
        self.s = None
        self.u = None
        self.v = None
        # plt.close('all')

    def update_theta(self):

        self.neural_net.zero_grad()

        for i in range(3):
            CE_loss = self.CEloss_criterion(self.limage_output, self.lmask.squeeze(1))
            unlabled_loss = self.p_u / 2 * (F.softmax(self.uimage_output, dim=1)[:, 1] + torch.from_numpy(-self.gamma+self.u).float().cuda()).norm(p=2) ** 2 \
                            + self.p_v /2 *(F.softmax(self.uimage_output,dim=1)[:,1] + torch.from_numpy(-self.s+self.v).float().cuda()).norm(p=2) ** 2

            loss = CE_loss + unlabled_loss
            # loss = unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            print('loss:', loss.item())

            self.uimage_forward(self.uimage, self.umask)
            self.limage_forward(self.limage,self.lmask)

    def set_boundary_term(self, g, nodeids, img, lumda, sigma):
        img = img.squeeze().cpu().data.numpy()
        pad_im = np.pad(img, ((0, 0), (1, 1)), 'constant', constant_values=0)
        weights = np.zeros((img.shape))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                weights[i, j] = lumda * np.exp((-1 / sigma) * np.abs(pad_im[i, j] - pad_im[i, j + 1]))
        structure = np.zeros((3, 3))
        structure[1, 2] = 1
        g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)

        pad_im = np.pad(img, ((1, 1), (0, 0)), 'constant', constant_values=0)
        weights = np.zeros((img.shape))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                weights[i, j] = lumda * np.exp((-1 / sigma) * np.abs(pad_im[i, j] - pad_im[i + 1, j]))
        structure = np.zeros((3, 3))
        structure[2, 1] = 1
        g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)

        return g

    def update_gamma(self):

        unary_term_gamma_1 = np.multiply(
            (0.5 - (F.softmax(self.uimage_output, dim=1).cpu().data.numpy()[:, 1, :, :] + self.u)),
            1)

        unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
        new_gamma = np.zeros(self.gamma.shape)
        g = maxflow.Graph[float](0, 0)
        i = 0
        # Add the nodes.
        nodeids = g.add_grid_nodes(list(self.gamma.shape)[1:])
        # Add edges with the same capacities.

        # g.add_grid_edges(nodeids, neighbor_term)
        g = self.set_boundary_term(g, nodeids, self.uimage, lumda=self.lamda, sigma=self.sigma)

        # Add the terminal edges.
        g.add_grid_tedges(nodeids, (unary_term_gamma_0[i]).squeeze(),
                          (unary_term_gamma_1[i]).squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids) * 1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_gamma[i] = np.int_(np.logical_not(sgm))
        # g.reset()
        self.gamma = new_gamma


    def update_s(self):
        a = 0.5 - (F.softmax(self.uimage_output,1)[:,1].cpu().data.numpy().squeeze() + self.v)
        original_shape = a.shape
        a_ = np.sort(a.ravel())
        useful_pixel_number = (a<0).sum()
        if self.lowbound< useful_pixel_number and self.upbound > useful_pixel_number:
            self.s = ((a<0)*1.0).reshape(original_shape)
        if useful_pixel_number < self.lowbound:
            self.s = ((a<=a_[self.lowbound])*1).reshape(original_shape)
        if useful_pixel_number > self.upbound:
            self.s = ((a<=a_[self.upbound])*1).reshape(original_shape)

    def update_u(self):

        # new_u = self.u + (F.softmax(self.uimage_output, dim=1)[:, 1, :, :].cpu().data.numpy() - self.gamma)
        new_u = self.u + (self.heatmap2segmentation(self.uimage_output).cpu().data.numpy() - self.gamma)
        # print(np.array((self.heatmap2segmentation(self.uimage_output).cpu().data.numpy() - self.gamma).nonzero()).sum())
        assert new_u.shape == self.u.shape


        self.u = new_u

        pass

    def update(self, limage_pair, uimage_pair):
        [limage,lmask], [uimage,umask] = limage_pair,uimage_pair
        self.limage_forward(limage, lmask)
        self.uimage_forward(uimage, umask)
        # self.update_theta()
        self.update_gamma()
        self.update_s()
        self.update_u()

    def show_labeled_pair(self):
        fig = plt.figure(1, figsize=(32, 32))
        plt.clf()
        fig.suptitle("labeled data", fontsize=16)

        ax1 = fig.add_subplot(221)
        ax1.imshow(self.limage[0].cpu().data.numpy().squeeze())
        ax1.title.set_text('original image')

        ax2 = fig.add_subplot(222)
        ax2.imshow(self.lmask[0].cpu().data.numpy().squeeze())
        ax2.title.set_text('ground truth')

        ax3 = fig.add_subplot(223)
        ax3.imshow(F.softmax(self.limage_output, dim=1)[0][1].cpu().data.numpy())
        ax3.title.set_text('prediction of the probability')

        ax4 = fig.add_subplot(224)
        ax4.imshow(np.abs(
            self.lmask[0].cpu().data.numpy().squeeze() - F.softmax(self.limage_output, dim=1)[0][1].cpu().data.numpy()))
        ax4.title.set_text('difference')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def show_ublabel_image(self):
        fig = plt.figure(2, figsize=(8, 8))
        fig.suptitle("Unlabeled data", fontsize=16)

        ax1 = fig.add_subplot(221)
        ax1.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        ax1.title.set_text('original image')
        ax1.set_axis_off()

        ax2 = fig.add_subplot(222)
        ax2.cla()
        # ax1.imshow(self.uimage[0].cpu().data.numpy().squeeze(),cmap='gray')
        ax2.imshow(F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy(), vmin=0, vmax=1, cmap='gray')
        # ax2.contour(F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy(),level=(0.5,0.5),colors="red",alpha=0.5)
        ax2.title.set_text('probability prediction')
        ax2.text(0,0,'%.3f'%F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy().max())
        # ax2.set_cl)
        ax2.set_axis_off()

        ax3 = fig.add_subplot(223)
        ax3.clear()
        ax3.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # ax3.imshow(self.umask.squeeze().cpu().data.numpy(),cmap='gray')
        ax3.contour(self.umask.squeeze().cpu().data.numpy(), level=[0], colors="red", alpha=1, linewidth=0.001)
        ax3.title.set_text('ground truth mask')
        ax3.set_axis_off()

        ax4 = fig.add_subplot(224)
        ax4.clear()
        ax4.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # ax4.imshow(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(),cmap='gray')
        ax4.contour(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(), level=[0.5],
                    colors="red", alpha=1, linewidth=0.001)
        # ax2.contour(F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy(),level=(0.5,0.5),colors="red",alpha=0.5)
        ax4.title.set_text('prediction mask')
        ax4.set_axis_off()
        # plt.tight_layout()
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        plt.show(block=False)
        plt.pause(0.01)

    def show_gamma(self):
        plt.figure(3, figsize=(5, 5))
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.imshow(self.gamma[0])
        plt.contour(self.umask.squeeze().cpu().data.numpy(), level=[0], colors="black", alpha=1, linewidth=0.001)
        plt.contour(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.5, linewidth=0.001)
        plt.contour(self.s.squeeze(),level=[0],colors='blue',alpha = 0.5, linewidth = 0.001)

        plt.contour(self.gamma[0], level=[0], colors="red", alpha=0.3, linewidth=0.001)
        plt.title('Gamma')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show(block=False)
        plt.pause(0.01)

    def show_u(self):
        plt.figure(4, figsize=(5, 5))
        plt.clf()
        plt.title('Multipicator')
        plt.subplot(1, 1, 1)
        plt.imshow(np.abs(self.u.squeeze()))
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.01)


if __name__ == "__main__":
    net = UNet(num_classes=2)
    net_ = networks(net, 10, 100)
    for i in xrange(10):
        limage = torch.randn(1, 1, 256, 256)
        uimage = torch.randn(1, 1, 256, 256)
        lmask = torch.randint(0, 2, (1, 256, 256), dtype=torch.long)
        net_.update((limage, lmask), uimage)
