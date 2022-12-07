import random


from torch.autograd import Variable
import torch
import os
from PIL import Image
import numpy as np

from torchvision.utils import save_image


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return max(0, 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch))

def load_pretrain(root, iters, G_AB=None, G_BA=None, D_A=None, D_B=None,G_Glyphy=None, D_Glyphy=None, Cls_net = None):#Cls_net待实现
    # Load pretrained models
    if G_AB is not None:
        path = root + '/G_AB_%d.pth' % iters
        G_AB.load_state_dict(torch.load(path,map_location='cuda:0'))
        print("load G_AB parameters from %s" % path)
    if G_BA is not None:
        path = root + '/G_BA_%d.pth' % iters
        G_BA.load_state_dict(torch.load(path,map_location='cuda:0'))
        print("load G_BA parameters from %s" % path)
    if D_A is not None:
        path = root + '/D_A_%d.pth' % iters
        D_A.load_state_dict(torch.load(path,map_location='cuda:0'))
        print("load D_A parameters from %s" % path)
    if D_B is not None:
        path = root + '/D_B_%d.pth' % iters
        D_B.load_state_dict(torch.load(path,map_location='cuda:0'))
        print("load D_B parameters from %s" % path)
    if G_Glyphy is not None:
        path = root + '/G_Glyphy_%d.pth' % iters
        G_Glyphy.load_state_dict(torch.load(path,map_location='cuda:0'))
        print("load G_Glyphy parameters from %s" % path)
    if D_Glyphy is not None:
        path = root + '/D_Glyphy_%d.pth' % iters
        D_Glyphy.load_state_dict(torch.load(path,map_location='cuda:0'))
        print("load D_Glyphy parameters from %s" % path)
    if Cls_net is not None:
        #path = root + '/Cls_net_%d.pth' % iters
        path = '/home/hongxianghuang/pycharmproject/OBC_Recognition/Outputs/models/alexnet_OBC306/2021-03-23_19-14-58/model_step99999.pth'
        Cls_net.load_state_dict(torch.load(path,map_location='cuda:0'),False)
        print("load Cls_net parameters from %s" % path)

def save_model(root, iters, G_AB=None, G_BA=None, D_A=None, D_B=None, G_Glyphy=None, D_Glyphy=None, Cls_net=None):
    if G_AB is not None:
        path = root + '/G_AB_%d.pth' % iters
        torch.save(G_AB.state_dict(), path)
    if G_BA is not None:
        path = root + '/G_BA_%d.pth' % iters
        torch.save(G_BA.state_dict(), path)
    if D_A is not None:
        path = root + '/D_A_%d.pth' % iters
        torch.save(D_A.state_dict(), path)
    if D_B is not None:
        path = root + '/D_B_%d.pth' % iters
        torch.save(D_B.state_dict(), path)
    if G_Glyphy is not None:
        path = root + '/G_Glyphy_%d.pth' % iters
        torch.save(G_Glyphy.state_dict(), path)
    if D_Glyphy is not None:
        path = root + '/D_Glyphy_%d.pth' % iters
        torch.save(D_Glyphy.state_dict(), path)
    if Cls_net is not None:
        path = root + '/Cls_net_%d.pth' % iters
        torch.save(Cls_net.state_dict(), path)

def mask2cycle_weight(mask_A):
    mask_A = mask_A / 2 + 0.5  # unnormlize to 0~1  [bs, 1, 64, 64]
    mask_A = torch.where(mask_A < 0.5, torch.zeros_like(mask_A), torch.ones_like(mask_A))  # binary
    num_bg_pixel = (mask_A == 0).to(torch.float32).flatten(1).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(
        1)  # [bs,1,1,1]
    num_fg_pixel = (mask_A == 1).to(torch.float32).flatten(1).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(
        1)  # [bs,1,1,1]
    num_total_pixel = mask_A.size(2) * mask_A.size(3)  # 64*64
    c, _ = torch.max(torch.stack([num_bg_pixel, num_fg_pixel], dim=0), dim=0, keepdim=False)
    w_fg = 2 *c / num_fg_pixel # num_bg_pixel / num_total_pixel
    w_bg = c / num_bg_pixel # num_fg_pixel / num_total_pixel
    w_fg, mask_A = torch.broadcast_tensors(w_fg, mask_A)
    w_bg, mask_A = torch.broadcast_tensors(w_bg, mask_A)
    cycle_weight = torch.where(mask_A == 1, w_fg, w_bg)
    return cycle_weight


def agtgan_sample_images(G_AB, G_BA,G_Glyphy,val_dataloader, i, root):
    G_AB.eval()
    G_BA.eval()
    G_Glyphy.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))
    fake_G_Glyphy = G_Glyphy(real_A)[0]
    img_sample = torch.cat((real_A.data, fake_G_Glyphy.data,G_AB(fake_G_Glyphy)[0].data,
                            real_B.data,G_BA(real_B)[0].data), 0)
    save_image(img_sample, os.path.join(root, "%06d.png" % i), nrow=5, normalize=True)
    G_AB.train()
    G_BA.train()
    G_Glyphy.train()

def sk_au_sample_images(G_Glyphy, G_BA, val_dataloader, i, root):
    G_BA.eval()
    G_Glyphy.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    # fake_B = G_AB(real_A)
    real_B = Variable(imgs['B'].type(Tensor))
    # fake_A = G_BA(real_B)
    img_sample = torch.cat((real_A.data, G_Glyphy(real_A)[0].data, G_Glyphy(real_A)[0].data,
                            real_B.data, G_BA(real_B)[0].data), 0)
    save_image(img_sample, os.path.join(root, "%06d.png" % i), nrow=5, normalize=True)
    G_BA.train()
    G_Glyphy.train()

def e2e_sample_images(G_AB, G_BA, G_Glyphy, val_dataloader, i, root):
    G_AB.eval()
    G_BA.eval()
    G_Glyphy.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))
    fake_Glyphy = G_Glyphy(real_A)[0].data
    img_sample = torch.cat((real_A.data, fake_Glyphy, G_AB(fake_Glyphy)[0].data,
                            real_B.data, G_BA(real_B)[0].data), 0)
    save_image(img_sample, os.path.join(root, "%06d.png" % i), nrow=5, normalize=True)
    G_AB.train()
    G_BA.train()
    G_Glyphy.train()

def to_binary(transformed_x):
    transformed_x = torch.threshold(transformed_x, 0, -1)
    transformed_x = -transformed_x
    transformed_x = torch.threshold(transformed_x, 0, -1)
    transformed_x = -transformed_x
    return transformed_x