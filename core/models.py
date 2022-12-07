import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core.config import cfg
from core.tps_stn.grid_sample import grid_sample
from core.tps_stn.tps_grid_gen import TPSGridGen
from utils.utils import *
import torchvision.models as models
l1_loss = torch.nn.L1Loss()
sigmoid = torch.nn.Sigmoid()



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of [Num_triplets, 3], each triplet indices to corners of triangle
    :param src_points: Tensor array of [Num_points, 2], each (x, y) point refer to a landmark for source image
    :param dst_points: Tensor array of [Num_points, 2], each (x, y) point refer to a landmark for source image
    :returns: [Num_triplets, 2, 3] affine matrix transformation for a triangle
    """
    ones = torch.ones([1, 3], dtype=src_points.dtype, device=src_points.device)
    mats = torch.zeros((len(vertices), 2, 3), device=src_points.device, dtype=torch.float32)
    for i, tri_indices in enumerate(vertices):
        src_tri = torch.cat((src_points[tri_indices, :].T, ones), dim=0)
        dst_tri = torch.cat((dst_points[tri_indices, :].T, ones), dim=0)
        mat = src_tri.matmul(dst_tri.inverse())[:2, :]
        mats[i] = mat
    return mats

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),  
                    nn.Conv2d(in_channels, 64, 7),
                    nn.InstanceNorm2d(64),  #
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(3):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        # for _ in range(2):
        model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) ]
        self.def1 = nn.Sequential(*model)

        in_features = out_features
        out_features = in_features//2
        second_deconv_block = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) ]
        self.def2 = nn.Sequential(*second_deconv_block)

        in_features = out_features
        out_features = in_features//2
        third_deconv_block = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) ]
        self.def3 = nn.Sequential(*third_deconv_block)

        in_features = out_features
        out_features = in_features//2
        # Output layer
        output_block = [nn.ReflectionPad2d(3),
                        nn.Conv2d(64, out_channels, 7),
                        nn.Tanh()]
        self.output_block = nn.Sequential(*output_block)


    def forward(self, x):
        def1 = self.def1(x)
        def2 = self.def2(def1)
        def3 = self.def3(def2)
        output = self.output_block(def3) #64x1x64x64 b,n,h,w
        return output, def1, def2, def3

##############################
#        Discriminator
##############################

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
            *discriminator_block(256, 512),
        )
        self.prob = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),  # 左右上下
            nn.Conv2d(512, 1, 4, padding=1)
            #nn.Conv2d(256,1,4,padding=1)
        )

        self.cls = nn.Sequential(
            nn.Linear(512, cls_num)
        )
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, img):
        features = self.features(img)
        output = self.prob(features)  
        return output, features


##############################
#        SkeletonNet
##############################

class SkeletonAugNet(nn.Module):
    def __init__(self, grid_h, grid_w, bs, z_en, c_cost_en):
        super(SkeletonAugNet, self).__init__()
        self.grid_generator = GridGenerator(grid_h, grid_w, bs, z_en, c_cost_en)
        self.smooth = SmoothLayer()
        self.sk_flag = True


    # (Spatial transformer network forward function)
    def forward(self, x):
        n,c,h,w = x.size()
        tps_grid, affine_grid, z_rec_loss, tps_diff, affine_diff, i_rec_loss = self.grid_generator(x)
        # tps warp
        transformed_x_ = grid_sample(x, tps_grid, padding_mode='border')
        # affine warp
        transformed_x_ = F.grid_sample(transformed_x_, affine_grid, padding_mode='border')
        transformed_x = self.smooth(transformed_x_)
        if self.training:
            aux_loss = l1_loss(transformed_x_, transformed_x)
            aux_loss.backward(retain_graph=True)
        return transformed_x, z_rec_loss, tps_diff, affine_diff, i_rec_loss

class SmoothLayer(nn.Module):
    def __init__(self):
        super(SmoothLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv3 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return F.tanh(x3)

"""预测TPS参数和仿射变换参数"""
class AugBackbone(nn.Module):
    def __init__(self, bs=64, num_output=1024, z_en=False, c_cost_en=False, image_rec_en=True): 
        super(AugBackbone, self).__init__()
        self.z_en = z_en
        self.image_rec_en = image_rec_en
        self.c_cost_en = c_cost_en
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),#320/2=160
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),#160/2=80
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),#20/2 =10
            nn.Conv2d(64, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(True),#16*10*10 = 1600
        )
        # 8*8*16
        self.avgpool1 = nn.AdaptiveAvgPool2d((8, 8))
        self.avgpool2 = nn.AdaptiveAvgPool2d((28, 28))
        self.fc = nn.Sequential(nn.Linear(1024, 1024),nn.BatchNorm1d(1024),nn.ReLU(True),
                                nn.Linear(1024, 1024),nn.BatchNorm1d(1024),nn.ReLU(True),
                                nn.Linear(1024, 1024),nn.BatchNorm1d(1024),nn.ReLU(True),
                                nn.Linear(1024, num_output))
        if z_en:
            self.z_encoder = nn.Sequential(nn.Linear(num_output+2, 1024),nn.ReLU(True),
                                    nn.Linear(1024, 1024),nn.ReLU(True),
                                    nn.Linear(1024, 1024),nn.ReLU(True),
                                    nn.Linear(1024, 1024))
        if image_rec_en:
            self.image_rec_fc = nn.Sequential(nn.Linear(num_output+2, 1024),nn.ReLU(True),
                                    nn.Linear(1024, 1024),nn.ReLU(True),
                                    nn.Linear(1024, 1024),nn.ReLU(True),
                                    nn.Linear(1024, 1024))
            self.image_rec_decnn = nn.Sequential(

                nn.Conv2d(16, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1), nn.Tanh()
            )


        self.pair_ids = list(itertools.combinations(range(bs),2))

    def forward(self, x):
        hidden = self.cnn(x)   #  [b, 16, grid, gird]
        #hidden = self.avgpool1(hidden)
        n,c,h,w = hidden.size()
        hidden = hidden.view(n, c*h*w)  # [b, 1024] ->[b, 1600]
        pred_aux = None
        z_rec_loss = None
        image_rec_loss = None
        if self.training:
            if self.z_en:
                hidden_power = torch.sqrt(torch.mean(hidden ** 2, dim=1)+1e-6)
                z = hidden_power.view(-1, 1) * torch.randn(hidden.size(), dtype=hidden.dtype, layout=hidden.layout, device=hidden.device) 
                pred = self.fc(hidden + z)  # [b, 128]
                z2 = hidden_power.view(-1, 1) * torch.randn(hidden.size(), dtype=hidden.dtype, layout=hidden.layout, device=hidden.device)
                pred_aux = self.fc(hidden + z2) 
                #
                pred_bound = torch.cat(self.bound_output(pred), dim=1)
                z_rec = self.z_encoder(pred_bound)
                z_rec_loss = l1_loss(z, z_rec)
            else:
                pred = self.fc(hidden)
                z_rec_loss = None
            if self.c_cost_en:
                pass
                # pairs = pred[self.pair_ids, ...]  # [1024,2,1024]
                # tps_diff = l1_loss(pairs[:, 0, :-4], pairs[:, 1, :-4])
                # affine_diff = l1_loss(pairs[:, 0, -4:], pairs[:, 1, -4:])
            else:
                tps_diff = None
                affine_diff = None
            if self.image_rec_en:
                fc_rec = self.image_rec_fc(pred_bound)  # [b, 128]
                #fc_rec = fc_rec.view(n, c, h, w)
                fc_rec = fc_rec.view(n, c, h, w)
                #fc_rec = self.avgpool2(fc_rec)
                image_rec = self.image_rec_decnn(fc_rec)
                image_rec_loss = l1_loss(x, image_rec)
                
            else:
                image_rec_loss = None
        else:
            if self.z_en:
                hidden_power = torch.sqrt(torch.mean(hidden ** 2, dim=1)+1e-6)
                z = hidden_power.view(-1, 1) * torch.randn(hidden.size(), dtype=hidden.dtype, layout=hidden.layout, device=hidden.device)
                pred = self.fc(hidden + z)  # [b, 128]
            z_rec_loss = None
            tps_diff = None
            affine_diff = None
            image_rec_loss = None
        return self.bound_output(pred), z_rec_loss, image_rec_loss, self.bound_output(pred_aux)

    def bound_output(self, pred):
        tps_deltas = None
        affine_theta = None
        if pred is not None:
            tps_deltas = pred[:, :-4]
            affine_theta = pred[:, -4:]
            tps_deltas = F.tanh(tps_deltas) / 18. 
            theta, tx, ty, scale = torch.split(affine_theta, split_size_or_sections=1, dim=1)
            theta = 3.1415926 / 8 / 2 * torch.tanh(theta) 
            tx = torch.tanh(tx) / 5 
            ty = torch.tanh(ty) / 5 
            scale = 1 + 0.15 * torch.tanh(scale)  
            affine_theta = torch.cat([scale * torch.cos(theta), -scale * torch.sin(theta), tx,
                                      scale * torch.sin(theta), scale * torch.cos(theta), ty], dim=1)
        return tps_deltas, affine_theta


class GridGenerator(nn.Module):

    def __init__(self, grid_h, grid_w, bs, z_en=False, c_cost_en=False):
        super(GridGenerator, self).__init__()
        n_grid = grid_h * grid_w
        # backbone net
        self.backbone = AugBackbone(bs=bs, num_output=n_grid * 2 + 4, z_en=z_en, c_cost_en=c_cost_en) 
        self.c_cost_en = c_cost_en
        self.pair_ids = list(itertools.combinations(range(bs), 2))
        # generate basic sampling grid
        xs = (torch.arange(0, grid_w)-(grid_w/2.)) / (grid_w/2.)
        ys = (torch.arange(0, grid_h)-(grid_h/2.)) / (grid_h/2.)
        xs, ys = torch.meshgrid(xs,ys)
        basic_grid = torch.cat([ys.reshape(n_grid, 1), xs.reshape(n_grid, 1)], dim=1) 
        self.basic_grid = basic_grid.reshape(1, n_grid, 2)

        grid_height = grid_h
        grid_width = grid_w
        r1 = 0.9
        r2 = 0.9
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
                                                                    np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
                                                                    )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)
        self.tps = TPSGridGen(n_grid, n_grid, target_control_points)
        # ratation
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        self.backbone.fc[-1].bias.data.zero_()
        self.backbone.fc[-1].weight.data.zero_()
        # init affine scale to one
        self.backbone.fc[-1].bias.data[-1] = 1  # [128+6,]


    def forward(self, x):
        batch_size = x.size(0)
        (tps_deltas, affine_theta), z_rec_loss, i_rec_loss, (tps_deltas_aux, affine_theta_aux) = self.backbone(x)
        # diff loss
        tps_diff = None
        affine_diff = None
        if self.c_cost_en and self.training:
            tps_diff = l1_loss(tps_deltas, tps_deltas_aux)
            affine_diff = l1_loss(affine_theta, affine_theta_aux)
        tsp_points = self.basic_grid.to(tps_deltas.device) + tps_deltas.view(batch_size, -1, 2) #(1,ngrid,2)
        # tps grid
        source_coordinate = self.tps(tsp_points)
        tps_grid = source_coordinate.view(batch_size, 64, 64, 2)
        # affine grid
        affine_theta = affine_theta.view(-1, 2, 3)
        affine_grid = F.affine_grid(affine_theta, x.size())
        return tps_grid, affine_grid, z_rec_loss, tps_diff, affine_diff, i_rec_loss

class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_out, self).__init__()
        #vgg = models.vgg19(pretrained=True).to(device)  # .cuda()
        # vgg.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))
        vgg = models.vgg19(pretrained=True).to('cuda:%d'%(cfg.CUDANUM.FIRST))
        vgg.features[0] = nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)

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
        #self.vgg = Vgg19_out().to(device)
        self.vgg = Vgg19_out().to('cuda:%d'%(cfg.CUDANUM.FIRST))
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
        #self.vgg = Vgg19_out().to(device)
        self.vgg = Vgg19_out().to('cuda:%d'%(cfg.CUDANUM.FIRST))
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
            #print(iter + 1, self.criterion(x_fea, y_fea.detach()), x_fea.size())
            loss += self.criterion(x_fea, y_fea.detach())
        return loss

class Aadaptor(nn.Module):
    pass

if __name__ == '__main__':
    model = Discriminator()
    x = torch.rand(1,3,64,64)
    y = model(x)