import os,six,glob,sys,random
import numpy as np
from PIL import Image
from core.models import GeneratorResNet,SkeletonAugNet
import torch,io
import torchvision.transforms as transforms

def to_white_on_black(img_np):
    # assume that bg pixel num > fg pixel num, and the pixel vals are normlized to 0~255
    thresh = (img_np.max() + img_np.min()) / 2.0
    bigger_val_num = (img_np > thresh).astype(np.float32).sum()
    smaller_val_num = img_np.size - bigger_val_num
    if bigger_val_num < smaller_val_num:
        return img_np
    else:
        return 255 - img_np

transforms_ = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                 transforms.Resize((64,64),Image.BICUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5],[0.5])])

destyle_vis_root = 'to_white_on_black_vis'
A_root = 'data/oracle/train/A'

input_folders = []
'''
valid_classes = os.listdir(B_path)
for d in valid_classes:
     if os.path.exists(os.path.join(A_path, d)):
          input_folders.append(os.path.join(A_path, d))
'''
dict_txt_r = '/home/hongxianghuang/pycharmproject/OBC_net/aux_files/destyle_class_list.txt'
valid_classes = open(dict_txt_r).read().splitlines()
for d in valid_classes:
     if os.path.exists(os.path.join(A_root, d)):
          input_folders.append(os.path.join(A_root, d))

#target_num = 10
sum = 0
for folder in input_folders:
    B_files = glob.glob(os.path.join(folder,'*.bmp'))
    for n,f in enumerate(B_files):
        #if n < 10:
            #i = 0
            #while i < target_num:

                img_B_pil = Image.open(f)
                img_B_np = np.array(img_B_pil)
                img_B_np = img_B_np / img_B_np.max().astype(np.float32) * 255
                img_B_np = to_white_on_black(img_B_np)
                item_B = transforms_(Image.fromarray(np.uint8(img_B_np)))
                img_tensor = item_B.unsqueeze(0).cuda()
                ndarr = ((img_tensor[0][0].cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
                im = Image.fromarray(ndarr)
                im_vis_path = os.path.join(destyle_vis_root, 'vis', folder[-6:],
                                           '{}_{}.jpg'.format(folder[-6:], n))
                os.makedirs(os.path.dirname(im_vis_path), exist_ok=True)
                im.save(im_vis_path, format='jpeg', quality=90)
                sum += n
                print('generated : %d' % sum)
