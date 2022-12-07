import glob
import random
import os
from skimage.color.adapt_rgb import adapt_rgb, each_channel

import cv2 as cv
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import torch
import numpy as np

def to_white_on_black(img_np):
    # assume that bg pixel num > fg pixel num, and the pixel vals are normlized to 0~255
    thresh = (img_np.max() + img_np.min()) / 2.0
    bigger_val_num = (img_np > thresh).astype(np.float32).sum()
    smaller_val_num = img_np.size - bigger_val_num
    if bigger_val_num > smaller_val_num:
        return 255 - img_np
    else:
        return img_np


class OBC_KP_Dataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', filter_A=True):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*/*.*'))
        # filter the categories do not exist in B
        if filter_A:
            self.files_A = []
            valid_c_root = os.listdir(os.path.join(root, '%s/B' % mode))
            for d in valid_c_root:
                self.files_A.extend(glob.glob(os.path.join(root, '%s/A' % mode, d, "*.bmp")))
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*/*.bmp'))
        self.A_bboxes = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*/*.pkl'))
        self.files_A = sorted(self.files_A)

    def __getitem__(self, index):
        if self.unaligned:
            img_B_pil = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            img_B_pil = Image.open(self.files_B[index % len(self.files_B)])
        item_B = self.transform(img_B_pil)
        img_A_pil = Image.open(self.files_A[index % len(self.files_A)])
        img_A_np = np.array(img_A_pil)
        # 添加黑色边框，避免torch的border插值去复制出现异常边缘
#         img_A_np = np.insert(img_A_np, 0, values=0, axis=0)
#         img_A_np = np.insert(img_A_np, 1, values=0, axis=0)
#         img_A_np = np.insert(img_A_np, 0, values=0, axis=1)
#         img_A_np = np.insert(img_A_np, 1, values=0, axis=1)
        img_A_np = img_A_np/img_A_np.max().astype(np.float32)*255  # renormlize to 0~255
        img_A_np  = to_white_on_black(img_A_np)
        item_A = self.transform(Image.fromarray(img_A_np))
        mask_A = item_A.clone()
        # read OBC keypoints
        '''
        A_bboxes = pickle.load(open(self.A_bboxes[index % len(self.files_A)], 'rb'))
        A_bboxes = torch.Tensor(A_bboxes)
        kps_x = (A_bboxes[:, 0] + A_bboxes[:, 2]) / 2.
        kps_y = (A_bboxes[:, 1] + A_bboxes[:, 3]) / 2.
        A_kps = torch.stack((kps_x, kps_y), dim=1)
        A_kps[:, 0] /= float(img_A_pil.width)   # normize x to 0~1
        A_kps[:, 1] /= float(img_A_pil.height)
        num_kps = A_kps.size(0)
        if num_kps < 100:
            A_kps = torch.cat((A_kps, torch.zeros((100-num_kps, 2))), dim=0)  #为了能支持bs>1，保证形状一致，补到[100,2]
        else:
            A_kps = A_kps[:100]
'''
        

        #return {'A': item_A, 'B': item_B, 'mask_A': mask_A, 'keypoints': A_kps, 'num_kps': num_kps}
        return {'A': item_A, 'B': item_B, 'mask_A': mask_A}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class Cuneiform_Dataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*/*.jpg'))
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*/*.png'))

    def __getitem__(self, index):
        if self.unaligned:
            img_B_pil = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            img_B_pil = Image.open(self.files_B[index % len(self.files_B)])
        # img_B_pil = cuneiform_prepare(img_B_pil)
        item_B = self.transform(img_B_pil)
        img_A_pil = Image.open(self.files_A[index % len(self.files_A)])
        img_A_np = np.array(img_A_pil)
        img_A_np = img_A_np/img_A_np.max().astype(np.float32)*255  # renormlize to 0~255
        img_A_np_mask  = to_white_on_black(img_A_np)
        item_A = self.transform(Image.fromarray(img_A_np.astype(np.uint8)))
        mask_A = self.transform(Image.fromarray(img_A_np_mask.astype(np.uint8)))
        return {'A': item_A, 'B': item_B, 'mask_A': mask_A}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def update_gc(inputimg, i, j):
    #对二维数组（图片）进行操作
    #数组不取到最后一个像素点的原因，可以参考我后面画的图，因为在步长为2的情况下是取不到最后一个点的
    inputimg_ij = inputimg[i:-1:2, j:-1:2]
    #求di（根据公式写程序，基点是[i:-1:2, j:-1:2]）
    #d1中[i - 1:-2:2, j:-1:2] 也可以参考后面画的图，因为它是基点的左边一个点，是模块中心点正左方的点，所以取不到倒数第二个点
    d1 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i + 1::2, j:-1:2])/2.0 - inputimg[i:-1:2, j:-1:2]
    d2 = (inputimg[i:-1:2, j - 1:-2:2] + inputimg[i:-1:2, j + 1::2])/2.0 - inputimg[i:-1:2, j:-1:2]
    d3 = (inputimg[i - 1:-2:2, j - 1:-2:2] + inputimg[i + 1::2, j + 1::2])/2.0  - inputimg[i:-1:2, j:-1:2]
    d4 = (inputimg[i - 1:-2:2, j + 1::2] + inputimg[i + 1::2, j - 1:-2:2])/2.0 - inputimg[i:-1:2, j:-1:2]
    # d5 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i:-1:2, j - 1:-2:2] -inputimg[i - 1:-2:2, j - 1:-2:2]) - inputimg[i:-1:2, j:-1:2]
    # d6 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i:-1:2, j + 1::2] - inputimg[i - 1:-2:2,j + 1::2])- inputimg[i:-1:2, j:-1:2]
    # d7 = (inputimg[i:-1:2, j- 1:-2:2] + inputimg[i + 1::2, j:-1:2] - inputimg[i + 1::2, j - 1:-2:2])- inputimg[i:-1:2, j:-1:2]
    # d8 = (inputimg[i:-1:2, j + 1::2] + inputimg[i + 1::2, j:-1:2] - inputimg[i + 1::2, j + 1::2])- inputimg[i:-1:2, j:-1:2]
    
    d5 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i:-1:2, j - 1:-2:2]/3.0 - inputimg[i - 1:-2:2, j - 1:-2:2]) - inputimg[i:-1:2, j:-1:2]
    d6 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i:-1:2, j + 1::2]/3.0 - inputimg[i - 1:-2:2,j + 1::2]) - inputimg[i:-1:2, j:-1:2]
    d7 = (inputimg[i:-1:2, j - 1:-2:2] + inputimg[i + 1::2, j:-1:2]/3.0 - inputimg[i + 1::2, j - 1:-2:2]) - inputimg[i:-1:2, j:-1:2]
    d8 = (inputimg[i:-1:2, j + 1::2] + inputimg[i + 1::2, j:-1:2]/3.0 - inputimg[i + 1::2, j + 1::2]) - inputimg[i:-1:2, j:-1:2]
    #找到di最小值并赋值给d
    #条件满足则赋1不满足赋0，若d2<d1，则np.abs(d2) < np.abs(d1)为1，np.abs(d1) <= np.abs(d2)为0，d=d2
    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d5)) + d5 * (np.abs(d5) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d6)) + d6 * (np.abs(d6) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d7)) + d7 * (np.abs(d7) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d8)) + d8 * (np.abs(d8) < np.abs(d))
    #list.append(d)
    inputimg_ij[...] +=d

updaterule = { 'gc': update_gc}
@adapt_rgb(each_channel)
def cf_filter(inputimg, filtertype, total_iter = 10, dtype = np.float32):
    #断言 如果不满足括号内条件系统报错并提示后一句
    assert(type(filtertype) is str), "input argument is not a string datatype!"
    #判断调用的滤波器是否在路由中
    assert(filtertype in updaterule.keys()), "filter type is not found!"
    #不对原图片进行修改  复制一份进行操作
    filteredimg = np.copy(inputimg.astype(dtype))
    #获取到函数名
    update = updaterule.get(filtertype)
    #遍历
    for iter_num in range(total_iter):
        update(filteredimg, 1, 1)
        update(filteredimg, 2, 2)
        update(filteredimg, 1, 2)
        update(filteredimg, 2, 1)
    return filteredimg

def cuneiform_prepare(im):
    arr=np.array(im)
    for i in range(10):
        result=cf_filter(arr,'gc',total_iter = 3)
        arr = result
    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5, 5))
    closeImg = cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel)
    img_norm = closeImg/255.0  #注意255.0得采用浮点数
    img_gamma = np.power(img_norm,0.4)*255.0
    out = img_gamma.astype(np.uint8)
    pil_im = Image.fromarray(out).convert('RGB')
    return pil_im