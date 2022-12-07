import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from core.models import *
from utils.datasets import *
from utils.utils import *
import torch
import time, datetime, sys
from core.config import cfg
import core.pytorch_ssim.pytorch_ssim as pytorch_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default="configs/OBCNET_e2e.yaml", help='path of the config file')


def train_iter(real_A, real_B, G_AB, G_BA, D_A, D_B,G_Glyphy,D_Glyphy,optimizer_G_Glyphy,optimizer_D_Glyphy ,optimizer_G,optimizer_D_A, optimizer_D_B, valid, fake, cycle_weight):

    # log losses
    metrics = {}
    # ------------------
    #  Train Generators
    # ------------------
    optimizer_G.zero_grad()
    optimizer_G_Glyphy.zero_grad()
    optimizer_smooth.zero_grad()

    # Identity loss
    # loss_id_A = criterion_identity(G_BA(real_A)[0], real_A)
    # loss_id_B = criterion_identity(G_AB(real_B)[0], real_B)
    # loss_identity = (loss_id_A + loss_id_B) / 2

    # GAN loss A and B
    fake_B = G_AB(real_A)[0]
    loss_GAN_AB = criterion_GAN(D_B(fake_B)[0], valid)

    fake_A = G_BA(real_B)[0]
    loss_GAN_BA = criterion_GAN(D_A(fake_A)[0], valid)
    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
    metrics['loss_GAN'] = loss_GAN.item()

    # G_Glyphy loss
    fake_Glyphy, z_rec_loss, tps_diff, affine_diff, i_rec_loss = G_Glyphy(real_A)
    similar = ssim_loss(real_A / 2 + 0.5, fake_Glyphy / 2 + 0.5)
    metrics['similar'] = similar.item()
    metrics['z_rec_loss'] = z_rec_loss.item()
    metrics['tps_diff'] = tps_diff.item()
    metrics['affine_diff'] = affine_diff.item()
    metrics['image_rec_loss'] = i_rec_loss.item()
    SNR = z_rec_loss / i_rec_loss  
    if SNR < 1/6:
        (-SNR).backward(retain_graph=True)
    if SNR > 6:
        SNR.backward(retain_graph=True)
    metrics['SNR'] = SNR.item()
    similar_loss = soomthL1(similar, similar_target.to(similar))
    tps_div_loss = -tps_diff
    affine_div_loss = -affine_diff
    loss_GAN_Glyphy = criterion_GAN(D_Glyphy(fake_Glyphy)[0], valid)
    metrics['loss_GAN_Glyphy'] = loss_GAN_Glyphy.item()
    loss_G_Glyphy = (lambda_G_Glyphy * loss_GAN_Glyphy + lambda_si * similar_loss + lambda_z * z_rec_loss + \
                    lambda_tps_div * tps_div_loss + lambda_affine_div * affine_div_loss + lambda_i * i_rec_loss) / \
                    (lambda_G_Glyphy+lambda_si+lambda_z+lambda_tps_div+lambda_affine_div+lambda_i)

    # Cycle loss A and B
    recov_A = G_BA(fake_B)[0]
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    loss_cycle_A = (cycle_weight * loss_cycle_A).mean()  # weight cycle loss

    recov_B = G_AB(fake_A)[0]
    loss_cycle_B = 0.5 * criterion_cycle(recov_B, real_B).mean()  # default
    loss_cycle_AB = (loss_cycle_A + loss_cycle_B) / 2
    metrics['loss_cycle_AB'] = loss_cycle_AB.item()

    # Total loss
    loss_G_Glyphy.backward(retain_graph=True) 
    loss_G = loss_GAN + lambda_cyc_AB * loss_cycle_AB
    loss_G.backward()
    optimizer_G_Glyphy.step()
    optimizer_smooth.step()
    optimizer_G.step()
    metrics['loss_G'] = loss_G.item()


    # -----------------------
    #  Train Discriminator A
    # -----------------------
    optimizer_D_A.zero_grad()
    # Real loss
    loss_real = criterion_GAN(D_A(real_A)[0], valid)

    # Fake loss (on batch of previously generated samples)
    fake_A_ = fake_A_buffer.push_and_pop(fake_A)
    loss_fake = criterion_GAN(D_A(fake_A_.detach())[0], fake)

    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2

    loss_D_A.backward()
    optimizer_D_A.step()

    # -----------------------
    #  Train Discriminator B
    # -----------------------
    optimizer_D_B.zero_grad()
    # Real loss
    loss_real = criterion_GAN(D_B(real_B)[0], valid)

    # Fake loss (on batch of previously generated samples)
    fake_B_1_ = fake_B_1_buffer.push_and_pop(fake_B)
    loss_fake1 = criterion_GAN(D_B(fake_B_1_.detach())[0], fake)


    fake_B_2_ = fake_B_2_buffer.push_and_pop(G_AB(fake_Glyphy)[0])
    loss_fake2 = criterion_GAN(D_B(fake_B_2_.detach())[0], fake)
    loss_fake = (loss_fake1 + loss_fake2) / 2

    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2
    loss_D_B.backward()
    optimizer_D_B.step()

    loss_D = (loss_D_A + loss_D_B) / 2
    metrics['loss_D'] = loss_D.item()

    # -----------------------
    #  Train Discriminator Glyphy
    # -----------------------
    optimizer_D_Glyphy.zero_grad()
    # Real loss
    loss_real = criterion_GAN(D_Glyphy(fake_A_.detach())[0], valid) 
    # Fake loss (on batch of previously generated samples)
    fake_Glyphy_ = fake_Glyphy_buffer.push_and_pop(fake_Glyphy)
    loss_fake = criterion_GAN(D_Glyphy(fake_Glyphy_.detach())[0], fake)
    # Total loss
    loss_D_Glyphy = (loss_real + loss_fake) / 2
    loss_D_Glyphy.backward()
    optimizer_D_Glyphy.step()
    metrics['loss_D_Glyphy'] = loss_D_Glyphy.item()
    return metrics

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    #################################################
    # Create tboard, sample and checkpoint directories
    #################################################
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cfg.merge_from_file(args.config_file)
    log_dir = os.path.join(cfg.OUTPUT_DIR, os.path.basename(args.config_file)[:-5], t)
    tboard_dir = os.path.join(log_dir, 'tboard')
    model_dir = os.path.join(log_dir, 'model')
    sample_dir = os.path.join(log_dir, 'sample')
    os.makedirs(tboard_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    tfb_summary = SummaryWriter(tboard_dir)
    print("Create log directory in {}".format(log_dir))

    #################################################
    # Create DataLoader
    #################################################
    # Image transformations
    transforms_ = [transforms.Grayscale(num_output_channels=1),
                   transforms.Resize((cfg.INPUT.IMAGE_HEIGHT, cfg.INPUT.IMAGE_WIDTH), Image.BICUBIC),
                   transforms.ToTensor(),
                   transforms.Normalize([0.5], [0.5])]
    # Training data loader
    if cfg.DATASETS.NAME == 'oracle':
        train_dataloader = DataLoader(OBC_KP_Dataset(cfg.DATASETS.DIR, transforms_=transforms_, unaligned=True),
                                batch_size=cfg.SOLVER.IMS_PER_BATCH*cfg.NUM_GPUS, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
        # Test data loader
        val_dataloader = DataLoader(OBC_KP_Dataset(cfg.DATASETS.DIR, transforms_=transforms_, unaligned=True, mode='test'),
                                batch_size=5, shuffle=True, num_workers=1)
    elif cfg.DATASETS.NAME == 'cuneiform':
        train_dataloader = DataLoader(Cuneiform_Dataset(cfg.DATASETS.DIR, transforms_=transforms_, unaligned=True),
                                batch_size=cfg.SOLVER.IMS_PER_BATCH*cfg.NUM_GPUS, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
        # Test data loader
        val_dataloader = DataLoader(Cuneiform_Dataset(cfg.DATASETS.DIR, transforms_=transforms_, unaligned=True, mode='test'),
                                batch_size=5, shuffle=True, num_workers=1)

    #################################################
    # Create Losses Criterion
    #################################################
    ssim_loss = pytorch_ssim.SSIM()
    soomthL1 = torch.nn.SmoothL1Loss()

    criterion_GAN = torch.nn.MSELoss()

    criterion_cycle = torch.nn.L1Loss(reduction='none')
    criterion_identity = torch.nn.L1Loss()
    # Loss weights
    lambda_cyc_AB = 10
    lambda_Gy = 1
    # lambda_id = 0.5 * lambda_cyc
    lambda_id = 0

    lambda_si = cfg.SOLVER.LAMBDA_S
    lambda_z = cfg.SOLVER.LAMBDA_Z
    lambda_i = cfg.SOLVER.LAMBDA_I
    lambda_tps_div = cfg.SOLVER.LAMBDA_T
    lambda_affine_div = cfg.SOLVER.LAMBDA_A
    lambda_G_Glyphy = cfg.SOLVER.LAMBDA_G_G

    # regression target
    similar_target = torch.Tensor([cfg.SOLVER.SIMILAR_TARGET])
    tps_diff_target = torch.Tensor([cfg.SOLVER.TPS_DIFF_TARGET])
    affine_diff_target = torch.Tensor([cfg.SOLVER.AFFINE_DIFF_TARGET])
    #################################################
    # Create Network
    #################################################
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, cfg.INPUT.IMAGE_HEIGHT // 2 ** 4, cfg.INPUT.IMAGE_HEIGHT // 2 ** 4)
    # Initialize generator and discriminator
    G_AB = GeneratorResNet(in_channels=cfg.INPUT.IMAGE_CHANNELS, out_channels=cfg.INPUT.IMAGE_CHANNELS,
                           res_blocks=cfg.MODEL.STYLE_TRANSFER.N_RESIDUAL)
    G_BA = GeneratorResNet(in_channels=cfg.INPUT.IMAGE_CHANNELS, out_channels=cfg.INPUT.IMAGE_CHANNELS,
                           res_blocks=cfg.MODEL.STYLE_TRANSFER.N_RESIDUAL)
    D_A = Discriminator(in_channels=cfg.INPUT.IMAGE_CHANNELS)
    D_B = Discriminator(in_channels=cfg.INPUT.IMAGE_CHANNELS)
    #  Initialize Glyphy Augmentation
    G_Glyphy = SkeletonAugNet(cfg.MODEL.SKELETON_AUG.GRID_H, cfg.MODEL.SKELETON_AUG.GRID_W,
                              cfg.SOLVER.IMS_PER_BATCH,
                              cfg.MODEL.SKELETON_AUG.Z_EN, cfg.MODEL.SKELETON_AUG.C_COST_EN)
    D_Glyphy = Discriminator(in_channels=cfg.INPUT.IMAGE_CHANNELS)
    if cfg.MODEL.PRETRAIN_ITER > 0:
        # Load pretrained models
        load_pretrain(cfg.MODEL.PRETRAIN_DIR, cfg.MODEL.PRETRAIN_ITER, G_AB, G_BA, D_A, D_B)
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

        G_Glyphy.apply(weights_init_normal)
        D_Glyphy.apply(weights_init_normal)
    # TODO multi-GPU supporting
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

        G_Glyphy = G_Glyphy.cuda()
        D_Glyphy = D_Glyphy.cuda()

    #################################################
    # Create Optimizers
    #################################################
    epoch = cfg.MODEL.PRETRAIN_ITER / len(train_dataloader)
    lr = cfg.SOLVER.BASE_LR 
    b1 = cfg.SOLVER.ADAM_B1
    b2 = cfg.SOLVER.ADAM_B2

    n_epochs = cfg.SOLVER.MAX_ITER / len(train_dataloader)   
    decay_epoch = cfg.SOLVER.DECAY_ITER / len(train_dataloader)
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    optimizer_G_Glyphy = torch.optim.Adam(G_Glyphy.grid_generator.parameters(), lr=0.1 * lr, betas=(b1, b2))
    optimizer_D_Glyphy = torch.optim.Adam(D_Glyphy.parameters(), lr=0.1 * lr, betas=(b1, b2))
    optimizer_smooth = torch.optim.Adam(G_Glyphy.smooth.parameters(), lr=0.1 * lr, betas=(b1, b2))
    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(n_epochs, epoch,
                                                                          decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(n_epochs, epoch,
                                                                            decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(n_epochs, epoch,
                                                                            decay_epoch).step)

    lr_scheduler_G_Glyphy = torch.optim.lr_scheduler.LambdaLR(optimizer_G_Glyphy, lr_lambda=LambdaLR(n_epochs, epoch, 
                                                                                                     decay_epoch).step)
    lr_scheduler_D_Glyphy = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Glyphy,
                                                              lr_lambda=LambdaLR(n_epochs / 2, epoch / 2,
                                                                                 decay_epoch / 2).step)

    lr_scheduler_smooth = torch.optim.lr_scheduler.LambdaLR(optimizer_smooth, lr_lambda=LambdaLR(n_epochs, epoch,
                                                                                                 decay_epoch).step)

    #################################################
    # Start Training
    #################################################
    # Buffers of previously generated samples
    fake_Glyphy_buffer = ReplayBuffer()
    fake_A_buffer = ReplayBuffer()
    fake_B_1_buffer = ReplayBuffer()
    fake_B_2_buffer = ReplayBuffer()
    i_train_dataloader = iter(train_dataloader)
    prev_time = time.time()
    for i in range(cfg.MODEL.PRETRAIN_ITER, cfg.SOLVER.MAX_ITER):
        try:
            batch = next(i_train_dataloader)
        except StopIteration: 
            i_train_dataloader = iter(train_dataloader)
            batch = next(i_train_dataloader)
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

        real_A = batch['A'].type(Tensor)
        real_B = batch['B'].type(Tensor)
        mask_A = batch['mask_A'].type(Tensor)
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        cycle_weight = mask2cycle_weight(mask_A)
        metrics = train_iter(real_A, real_B, G_AB, G_BA, D_A, D_B, G_Glyphy, D_Glyphy, optimizer_G_Glyphy, optimizer_D_Glyphy, optimizer_G, optimizer_D_A, optimizer_D_B, \
                             valid, fake, cycle_weight)

        # check period event
        if (i+1) % cfg.SOLVER.TEST_PERIOD == 0:
            agtgan_sample_images(G_AB, G_BA,G_Glyphy ,val_dataloader, i, sample_dir)
        if (i+1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            save_model(model_dir, i, G_AB, G_BA, D_A, D_B, G_Glyphy, D_Glyphy)
        # write summary
        tfb_summary.add_scalars("AGTGAN/losses", metrics, i)
        lrs = {"G_lr": optimizer_G.state_dict()['param_groups'][0]['lr'],
               "D_A_lr": optimizer_D_A.state_dict()['param_groups'][0]['lr'],
               "D_B_lr": optimizer_D_B.state_dict()['param_groups'][0]['lr'],
               "G_Glyphy_lr": optimizer_G_Glyphy.state_dict()['param_groups'][0]['lr'],
               "D_Glyphy_lr": optimizer_D_Glyphy.state_dict()['param_groups'][0]['lr'],
               "smooth_lr": optimizer_smooth.state_dict()['param_groups'][0]['lr'],
               }
        tfb_summary.add_scalars("AGTGAN/lr", lrs, i)
        terminal_log = "\r[Iter %d/%d] " % (i, cfg.SOLVER.MAX_ITER)
        iter_left = cfg.SOLVER.MAX_ITER - i
        time_left = datetime.timedelta(seconds=iter_left * (time.time() - prev_time))
        prev_time = time.time()
        for k, v in metrics.items():
            terminal_log += "[{}: {:.5f}] ".format(k, v)
        terminal_log += "[ETA: {}] ".format(time_left)
        sys.stdout.write(terminal_log)
