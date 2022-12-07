import lmdb, os, six, glob, sys, time, datetime, random
import numpy as np
from PIL import Image
from core.models import GeneratorResNet, SkeletonAugNet
import torch, io
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils.datasets import to_white_on_black

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def tensor2pil():
    pass

class Folder_Dataset(Dataset):
    def __init__(self, root, file_type="*.bmp", transforms_=None):
        self.files = glob.glob(os.path.join(root, file_type.lower()))
        self.files.extend(glob.glob(os.path.join(root, file_type.upper())))
        self.files = sorted(self.files)
        self.transform = transforms.Compose(transforms_)

    def __getitem__(self, index):
        img_A_pil = Image.open(self.files[index % len(self.files)])
        img_A_np = np.array(img_A_pil)
        img_A_np = img_A_np / img_A_np.max().astype(np.float32) * 255  # renormlize to 0~255
        img_A_np = to_white_on_black(img_A_np)
        item_A = self.transform(Image.fromarray(img_A_np))
        return item_A, os.path.basename(self.files[index % len(self.files)])[:-4]

    def __len__(self):
        return len(self.files)


transforms_ = [transforms.Grayscale(num_output_channels=1),
                   transforms.Resize((64, 64), Image.BICUBIC),
                   transforms.ToTensor(),
                   transforms.Normalize([0.5], [0.5])]

if __name__ == '__main__':
    # prepare path and aux configs for generation 
    G_AB_root = 'logs/original_OBCNET_e2e/20211111_201856/model'  
    G_AB_iters = list(range(24999, 30000, 1000))
    G_Glyphy_root = 'logs/original_OBCNET_e2e/20211111_201856/model'
    G_Glyphy_iters = list(range(24999, 30000, 1000))
    A_root = 'data/oracle/train/A'
    B_root = 'data/oracle/train/B'
    input_folders = []
    valid_classes = os.listdir(B_root)
    for d in valid_classes:
        if os.path.exists(os.path.join(A_root, d)):
            input_folders.append(os.path.join(A_root, d))
    target_num = 1000 
    output_path = 'results/syn_database_e2e_au1000_20211111_201856'
    os.makedirs(output_path, exist_ok=True)
    dict_txt = 'aux_files/dict.txt'
    names = open(dict_txt).read().splitlines()
    c_dict = {}
    for c, name in enumerate(names):
        c_dict.update({name: c})
    # init lmdb
    lmdb_path = os.path.join(output_path)
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    # load GA-SN parematers
    G_Glyphy = SkeletonAugNet(8, 8, 1, True, True).cuda()
    G_AB = GeneratorResNet(in_channels=1, out_channels=1, res_blocks=4).cuda()


    # starget generation
    cache = {}
    cls_dict = {}
    cnt = 0
    for folder in input_folders:
        dataset = Folder_Dataset(folder, "*.bmp", transforms_)
        if len(dataset) == 0:
            continue
        dataloader = DataLoader(dataset, num_workers=0, batch_size=64, shuffle=False, drop_last=False)
        cnt_inner = 0
        idataloader = iter(dataloader)
        while cnt_inner < target_num:
            prev_time = time.time()
            # random load a GAB parameters
            iters = random.choice(G_AB_iters)
            G_AB_path = os.path.join(G_AB_root, 'G_AB_{}.pth'.format(iters))
            G_Glyphy_path = os.path.join(G_Glyphy_root, 'G_Glyphy_{}.pth'.format(iters))
            G_AB.load_state_dict(torch.load(G_AB_path))
            G_Glyphy.load_state_dict(torch.load(G_Glyphy_path))
            G_AB = G_AB.cuda().eval()
            G_Glyphy = G_Glyphy.cuda().eval()
            try:
                imgs, fnames = next(idataloader)
            except StopIteration:
                idataloader = iter(dataloader)
                imgs, fnames = next(idataloader)
            real_A = imgs.cuda()
            fake_Glyphy = G_Glyphy(real_A)
            fake_Glyphy = fake_Glyphy[0].detach()
            fake_B = G_AB(fake_Glyphy)[0].detach()
            # save fake_B to jpeg
            class_name = folder.split('/')[-1]
            ndarrs = ((fake_B.cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)  # unnormalize -- > untensor
            for i, ndarr in enumerate(ndarrs):
                im = Image.fromarray(ndarr[0])
                im_vis_path = os.path.join(output_path, 'vis', class_name, '{}_{}.jpg'.format(fnames[i], cnt_inner+i))
                os.makedirs(os.path.dirname(im_vis_path), exist_ok=True)
                im.save(im_vis_path, format='jpeg', quality=90)
                # save fake_B to lmdb
                im_buff = io.BytesIO()
                im.save(im_buff, format='jpeg', quality=95)
                imageBin = im_buff.getvalue()
                label = str(c_dict[class_name]).encode('utf-8')
                imageKey = ('image-%09d' % cnt).encode('utf-8')
                labelKey = ('label-%09d' % cnt).encode('utf-8')
                cache[imageKey] = imageBin
                cache[labelKey] = label
                try:
                    cls_dict[label.decode()].append(cnt)
                except:
                    cls_dict[label.decode()] = [cnt]
                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print('Written %d' % (cnt))
                cnt += 1
                cnt_inner += 1
            # log processing
            batches_left = target_num * len(input_folders) - cnt
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write("\r[Processing %d/%d] ETA: %s" % (cnt, target_num*len(input_folders), time_left))
    nSamples = cnt
    cache['num-samples'.encode('utf-8')] = str(nSamples).encode('utf-8')
    writeCache(env, cache)