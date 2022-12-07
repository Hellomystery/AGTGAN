'''
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchsummary import summary
import os
import matplotlib.pyplot as plt

vgg_model = models.vgg16_bn(pretrained=True)
for p in vgg_model.parameters():
    p.requires_grad = False
#vgg_model.load_state_dict(torch.load('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'),strict=False)

# class perceptual_loss(torch.nn.Module):
#     def __init__(self,vgg_model):
#         super(perceptual_loss, self).__init__()
#         self.vgg_layers = vgg_model

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        # self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            # print("vgg_layers name:",name,module)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(
                zip(output_features, gt_features, self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature) * loss_weight)
        return sum(loss), output_features  # /len(loss)


# 输入的应该时feature_maps.shape = (H,W,Channels)
# 下图对relu1_2 进行了可视化，有64channels，拼了个了8*8的图
def visualize_feature_map(feature_maps):
    # 创建特征子图，创建叠加后的特征图
    # param feature_batch: 一个卷积层所有特征图
    # np.squeeze(feature_maps, axis=0)
    print("visualize_feature_map shape:{},dtype:{}".format(feature_maps.shape, feature_maps.dtype))
    num_maps = feature_maps.shape[2]
    feature_map_combination = []
    plt.figure(figsize=(8, 7))
    # 取出 featurn map 的数量，因为特征图数量很多，这里直接手动指定了。
    # num_pic = feature_map.shape[2]
    row, col = get_row_col(num_maps)
    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(0, num_maps):
        feature_map_split = feature_maps[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')

    plt.savefig('./rain_pair/relu1_2_feature_map.png')  # 保存图像到本地
    plt.show()
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from torchvision import models
import os, cv2
from core.config import cfg

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:%d'%(cfg.CUDANUM.SECOND))

class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_out, self).__init__()
        vgg = models.vgg19(pretrained=True).to(device)  # .cuda()
        # vgg.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))

        vgg.eval()
        vgg_pretrained_features = vgg.features
        # print(vgg_pretrained_features)
        self.requires_grad = requires_grad
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):  # (3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # (3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):  # (7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):  # (12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):  # (21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Perceptual_loss134(nn.Module):
    def __init__(self):
        super(Perceptual_loss134, self).__init__()
        self.vgg = Vgg19_out().to(device)
        self.criterion = nn.MSELoss()
        # self.weights = [1.0/2.6, 1.0/16, 1.0/3.7, 1.0/5.6, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = self.weights[0] * self.criterion(x_vgg[0], y_vgg[0].detach()) + \
               self.weights[2] * self.criterion(x_vgg[2], y_vgg[2].detach()) + \
               self.weights[3] * self.criterion(x_vgg[3], y_vgg[3].detach())
        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19_out().to(device)
        self.criterion = nn.MSELoss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.0
        for iter, (x_fea, y_fea) in enumerate(zip(x_vgg, y_vgg)):
            print(iter + 1, self.criterion(x_fea, y_fea.detach()), x_fea.size())
            loss += self.criterion(x_fea, y_fea.detach())
        return loss


if __name__ == "__main__":
    fea_save_path = "./feature_save/"
    if not os.path.exists(fea_save_path):
        os.mkdir(fea_save_path)
    img1 = np.array(cv2.imread("/home/mdisk/hhx/pycharmprojects/downloads/001004_syn_key_before_1.jpg")) / 255.0
    img2 = np.array(cv2.imread("/home/mdisk/hhx/pycharmprojects/downloads/001004_syn_key_before_2.jpg")) / 255.0
    #img1 = transforms.Resize(64,64,'bicubic')
    img1 = img1.transpose((2, 0, 1))
    img2 = img2.transpose((2, 0, 1))
    print(img1.shape, img2.shape)
    img1_torch = torch.unsqueeze(torch.from_numpy(img1), 0).to(device)
    img2_torch = torch.unsqueeze(torch.from_numpy(img2), 0).to(device)
    img1_torch = torch.as_tensor(img1_torch, dtype=torch.float32).to(device)
    img2_torch = torch.as_tensor(img2_torch, dtype=torch.float32).to(device)

    vgg_fea = Vgg19_out()
    img1_vggFea = vgg_fea(img1_torch)
    print(len(img1_vggFea), img1_vggFea[0].shape)

    total_perceptual_loss = VGGLoss()
    perceptual_loss134 = Perceptual_loss134()
    loss1 = total_perceptual_loss(img1_torch, img2_torch)
    loss2 = perceptual_loss134(img1_torch, img2_torch)
    print(loss1, loss2)