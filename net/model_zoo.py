import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import warnings
from core.config import cfg
from torch.autograd import Function, Variable
from typing import Optional, Any, Tuple

from net.net import *
from net.incepv4 import *

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Parameters:
            - **alpha** (float, optional): :math:`α`. Default: 1.0
            - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
            - **max_iters** (int, optional): :math:`N`. Default: 1000
            - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class DAClassifer(nn.Module):

    def __init__(self, num_classes=31):
        super(DAClassifer, self).__init__()
        if cfg.MODEL.BACKBONE.lower() == 'alexnet':
            model = AlexNet(num_classes)
            model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )
            self.sharedNet = model
            self.cls_fn = nn.Linear(4096, num_classes)
            self.domain_fn = nn.Sequential(nn.Linear(4096, 2048),AdversarialNetwork(in_feature=2048))

    def forward(self, data):
        data = self.sharedNet(data)
        clabel_pred = self.cls_fn(data)
        dlabel_pred = self.domain_fn(WarmStartGradientReverseLayer()(data))
        return clabel_pred, dlabel_pred

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature,1024)
        self.ad_layer2 = nn.Linear(1024,1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, cls_num=306):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            #*discriminator_block(256, 512),
        )
        self.prob = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            #nn.Conv2d(512, 1, 4, padding=1)
            nn.Conv2d(256,1,4,padding=1)
        )
        #self.cls = nn.Sequential(
        #    *discriminator_block(512, cls_num),
        #    nn.AdaptiveAvgPool2d((1, 1))
        #)
        self.cls = nn.Sequential(
            #nn.Linear(4*4*512, 1024),
            nn.Linear(8*8*256,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, cls_num)
        )

    def forward(self, img):
        features = self.features(img)
        #output = self.prob(features)
        #cls_prob = self.cls(features)#第一种cls
        #return cls_prob.view(-1,306)#第一种cls
        #cls_prob = self.cls(features.view(-1,4*4*512))
        cls_prob = self.cls(features.view(-1, 8 * 8 * 256))
        return cls_prob

def SymmetricClassifier(num_classes=12):
    model = Discriminator(cls_num=num_classes)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    return model


def AlexNet(num_classes=12):
    model = models.alexnet(pretrained=False)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),

    )
    return model

def VGG16(num_classes=12):
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def ResNet18(num_classes=12):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    return model

def ResNet34(num_classes=12):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    return model

def ResNet50(num_classes=12):
    model = models.resnet50(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(w[:,:1,:,:])
    model.fc = nn.Linear(2048, num_classes)
    return model

def ResNet101(num_classes=12):
    model=models.resnet101(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    return model

'''
class Classiferincep(nn.Module):
    def __init__(self,num_classes=1000):
        super(Classiferincep, self).__init__()
        if cfg.MODEL.BACKBONE.lower() == 'inceptionv4':
            model = InceptionV4(num_classes)
            model.features[0].conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            self.shareNet = model.features[:7]  #更改sharenet层数，同时改自适应池化
            self.secondNet = model.features[7:]
            self.avg_pool1 = nn.AdaptiveAvgPool2d((2, 2))
            self.avg_pool2 = model.avg_pool

            self.first_linear = nn.Linear(1536,num_classes) #与sharenet有关
            self.last_linear = model.last_linear

    def forward(self, x):
        x = self.shareNet(x)
        #x1 = self.avg_pool(x)
        #x1 = x1.view(x1.size(0), -1)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.first_linear(x1)  #浅层分支
        x2 = self.secondNet(x)
        x2 = self.avg_pool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.last_linear(x2)
        return x1, x2
        
'''

class Classiferincep(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classiferincep, self).__init__()
        if cfg.MODEL.BACKBONE.lower() == 'inceptionv4':
            model = InceptionV4(num_classes)
            model.features[0].conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            self.shareNet = model.features[:7]  # 更改sharenet层数，同时改自适应池化
            self.secondNet = model.features[7:]
            self.thirdNet = nn.Sequential(
                nn.Conv2d(384,192,kernel_size=(3, 3), stride=(1,1),padding=(1,1),bias=False),
                nn.BatchNorm2d(192),
                nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(True),

            )
            self.avg_pool1 = nn.AdaptiveAvgPool2d((8, 8))
            self.avg_pool2 = model.avg_pool

            self.first_linear = nn.Linear(3072, num_classes)  # 与sharenet有关
            self.last_linear = model.last_linear

    def forward(self, x):
        x = self.shareNet(x)
        # x1 = self.avg_pool(x)
        # x1 = x1.view(x1.size(0), -1)
        x1 = self.thirdNet(x)
        x1 = self.avg_pool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.first_linear(x1)  # 浅层分支
        x2 = self.secondNet(x)
        x2 = self.avg_pool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.last_linear(x2)
        return x1, x2


'''
class Classiferresnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classiferresnet, self).__init__()
        if cfg.MODEL.BACKBONE.lower() == 'resnet50':
            model = models.resnet50(pretrained=True)
            w = model.conv1.weight
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.conv1.weight = torch.nn.Parameter(w[:, :1, :, :])
            self.shareNet = nn.Sequential(*list(model.children())[:5])
            self.secondNet = nn.Sequential(*list(model.children())[5:-2])
            self.adaptivepool_1 = nn.AdaptiveAvgPool2d((4, 4))
            self.adaptivepool_2 = model.avgpool
            self.first_linear = nn.Linear(4096, num_classes)
            self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.shareNet(x)
        # x1 = self.avg_pool(x)
        # x1 = x1.view(x1.size(0), -1)
        x1 = self.adaptivepool_1(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.first_linear(x1)  # 浅层分支

        x2 = self.secondNet(x)
        x2 = self.adaptivepool_2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.last_linear(x2)
        return x1, x2
'''

'''
class Classiferresnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classiferresnet, self).__init__()
        if cfg.MODEL.BACKBONE.lower() == 'resnet50':
            model1 = models.resnet50(pretrained=True)
            model2 = models.alexnet(pretrained=True)
            w = model1.conv1.weight
            model1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model1.conv1.weight = torch.nn.Parameter(w[:, :1, :, :])
            self.shareNet = nn.Sequential(*list(model1.children())[:4])
            self.addNet1 = nn.Sequential(
                model2.features[3:],
                model2.avgpool,
            )
            self.addNet2 = model2.classifier[:-1]
            self.secondNet = nn.Sequential(*list(model1.children())[4:-2])
            self.adaptivepool_1 = nn.AdaptiveAvgPool2d((4, 4))
            self.adaptivepool_2 = model1.avgpool

            self.first_linear = nn.Linear(4096, num_classes)
            self.last_linear = nn.Linear(2048, num_classes)
            self.addNet_linear = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.shareNet(x)
        # x1 = self.avg_pool(x)
        # x1 = x1.view(x1.size(0), -1)
        #x1 = self.adaptivepool_1(x)
        x1 = self.addNet1(x)
        x1 = x1.view(x1.size(0), -1)  #x = torch.flatten(x, 1)
        x1 = self.addNet2(x1)
        x1 = self.addNet_linear(x1)  # 浅层分支

        x2 = self.secondNet(x)
        x2 = self.adaptivepool_2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.last_linear(x2)
        return x1, x2
'''
class Classiferresnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classiferresnet, self).__init__()
        if cfg.MODEL.BACKBONE.lower() == 'resnet50':
            model1 = models.resnet50(pretrained=True)
            model2 = models.alexnet(pretrained=True)
            w1 = model1.conv1.weight
            model1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model1.conv1.weight = torch.nn.Parameter(w1[:, :1, :, :])
            w2 = model2.features[0].weight
            model2.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            model2.features[0].weight = torch.nn.Parameter(w2[:, :1, :, :])

            self.Net1 = nn.Sequential(*list(model2.children())[:-1])
            self.Net1_linear = model2.classifier[:-1]
            self.Net2 = nn.Sequential(*list(model1.children())[:-1])

            self.first_linear = nn.Linear(4096, num_classes)
            self.second_linear = nn.Linear(2048, num_classes)


    def forward(self, x):
        x1 = self.Net1(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.Net1_linear(x1)
        x1 = self.first_linear(x1)

        x2 = self.Net2(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.second_linear(x2)
        return x1, x2

class Discriminator_combin(nn.Module):
    def __init__(self, in_channels=3, cls_num=306):
        super(Discriminator_combin, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        self.prob = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        #self.cls = nn.Sequential(
        #    *discriminator_block(512, cls_num),
        #    nn.AdaptiveAvgPool2d((1, 1))
        #)
        self.cls1 = nn.Sequential(
            nn.Linear(512, cls_num)
        )
        self.cls2 = nn.Sequential(
            nn.Linear(4*4*512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, cls_num)
        )
        '''
        self.cls2 = nn.Sequential(
        nn.Dropout(),
        nn.Linear(4 * 4 * 512, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, cls_num)
        )

        '''
    def forward(self, img):
        features = self.features(img)
        output = self.prob(features) #矩阵
        #cls_prob = self.cls(features)#第一种cls
        #return cls_prob.view(-1,306)#第一种cls
        cls_prob = self.cls2(features.view(-1,4*4*512)) #1行 n列 返回scalar
        return output, cls_prob

def SymmetricClassifier_v2(num_classes=12):
    model = Discriminator_combin(cls_num=num_classes)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    return model