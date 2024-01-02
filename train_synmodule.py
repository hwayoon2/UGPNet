import wandb
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import os, inspect, shutil, json
from image_tools import preprocess, postprocess, Lanczos_resizing, ganimage_preprocess
import torchvision.transforms as transforms
import cv2
import numpy as np
import math
from data_augmentation import RandomAugmentation
from dataset import *
import random
from models.resmodule import ResModule
from archs.RRDBNet import RRDBNetX8
from models.synmodule import SynModule
from loss.gan_loss import cal_adv_d_loss, regd, cal_adv_loss
import yaml
from collections import OrderedDict
from basicsr.archs.stylegan2_arch import StyleGAN2Discriminator
from loss.Lpips import LpipsLoss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    if flag:
        model.train()
    else:
        model.eval()

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--job_name', type=str, default='nafnet_deblur')
    parser.add_argument('--save_dir', type=str, default='./training/SynModule')

    # Dataset path
    parser.add_argument('--train_gt_dir', type=str, default='')
    parser.add_argument('--train_lq_dir', type=str, default='')
    parser.add_argument('--test_gt_dir', type=str, default='')
    parser.add_argument('--test_lq_dir', type=str, default='')

    # pretrained path
    parser.add_argument('--resmodule_path', type=str, default='./ckpt/UGPNet_with_NAFNet_deblur/resmodule_best_20000.pth')
    parser.add_argument('--generator_path', type=str, default='./ckpt/pretrained/StyleGAN2/net_g.pth')
    parser.add_argument('--discriminator_path', type=str, default='./ckpt/pretrained/StyleGAN2/net_d.pth')

    # Training options
    parser.add_argument('--batch_num', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--initial_lr', type=float, default=1e-4)
    parser.add_argument('--perceptual_weight', type=float, default=10)
    parser.add_argument('--pixelwise_weight', type=float, default=1)
    parser.add_argument('--adv_weight', type=float, default=0.3)
    parser.add_argument('--d_lr', type=float, default=2.5e-5)
    parser.add_argument('--spatial_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--type', type=str, default='nafnet_deblur')
    parser.add_argument('--final_batch_idx', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--church', action="store_true")
    parser.add_argument('--randomize_noise', action="store_true")

    return parser.parse_args()

def main():
    wandb.init(project="train_syn")

    args = parse_args()

    wandb.run.name = f'{args.job_name}'
    wandb.run.save()
    wandb.config.update(args)

    work_dir = args.save_dir
    job_name = args.job_name
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)

    # Set random seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Save current file and arguments
    current_file_path = inspect.getfile(inspect.currentframe())
    current_file_name = os.path.basename(current_file_path)
    shutil.copyfile(current_file_path, os.path.join(work_dir, job_name, current_file_name))
    with open(os.path.join(work_dir, job_name, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args.type)

    if "rrdb" in args.type:
        resmodule = RRDBNetX8(3, 3)
        resmodule.cuda().requires_grad_(False).eval()
        resmodule.load_state_dict(torch.load(args.resmodule_path)['params_ema'])
    else:
        resmodule = ResModule(args)
        resmodule.cuda().requires_grad_(False).eval()
        resmodule.load_state_dict(torch.load(args.resmodule_path))

    synmodule = SynModule(args)
    synmodule.cuda().requires_grad_(True).train()
    synmodule.load_generator_only()


    #set discriminator
    chan = 2 if args.church else 1
    size = 256 if args.church else 512
    discriminator = StyleGAN2Discriminator(size, channel_multiplier=chan).cuda().requires_grad_(True).train()
    discriminator.load_state_dict(torch.load(args.discriminator_path)['params'])

    # set optimizers
    optimizer = torch.optim.Adam(synmodule.parameters(), lr=args.initial_lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr)


    #set losses
    cal_perceptual_loss = LpipsLoss()
    cal_l1_loss = nn.L1Loss()

    #set data
    train_prefetcher, valid_prefetcher = load_dataset(args)

    augmentation = RandomAugmentation(scale_range=[7/8, 9/8], translation_xrange=128, translation_yrange=50, rotation_range=5)

    savelist=['3030','3077','3000','3001','3004','3006','3012','3022','3031','3048','3072']

    batch_idx = 0

    while(True):
        pbar = tqdm(total=len(train_prefetcher))
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:
            
            loss_dict = {}

            lq = batch_data['lq'].cuda()
            gt = batch_data['gt'].cuda()

            with torch.no_grad():
                x_res = resmodule(lq)
                x_res, opt = augmentation(x_res, batch_idx)
                gt , _ = augmentation(gt, batch_idx, opt)
            
            x_syn = synmodule(x_res)

            # discriminator
            requires_grad(discriminator, True)
            requires_grad(synmodule, False)
            
            fake = x_syn.detach()
            real = gt
            
            fake_d_pred = discriminator(fake)
            real_d_pred = discriminator(real)
            d_loss = cal_adv_d_loss(fake_d_pred, real_d_pred)

            loss_dict['d_loss'] = d_loss.item()
            loss_dict['fake_d_pred'] = fake_d_pred.detach().mean().item()
            loss_dict['real_d_pred'] = real_d_pred.detach().mean().item()

            optimizer_D.zero_grad()
            d_loss.backward()

            if (batch_idx % 10 == 0):
                # Regularization
                real.requires_grad = True 
                real_pred = discriminator(real)
                l_d_r1 = regd(real, real_pred)
                real = real.detach()
                l_d_r1.backward()

            optimizer_D.step()

            # synmodule (encoder + generator)
            requires_grad(synmodule, True)
            requires_grad(discriminator, False)

            pixelwise_loss = cal_l1_loss(gt, x_syn)
            loss_dict['pixelwise_loss'] = pixelwise_loss.item()

            lpips_loss = cal_perceptual_loss(gt, x_syn)
            loss_dict['lpips_loss'] = lpips_loss.item()

            fake_g_pred = discriminator(x_syn)
            adv_loss = cal_adv_loss(fake_g_pred)

            loss_dict['adv_loss'] = adv_loss.item()

            total_loss = args.pixelwise_weight*pixelwise_loss + args.perceptual_weight*lpips_loss+ args.adv_weight*adv_loss
            loss_dict['total_loss'] = total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 50==0:
                wandb.log(loss_dict)

            batch_data = train_prefetcher.next()

            batch_idx += 1
            pbar.update(1)


            if (batch_idx % args.save_interval == 0) or (batch_idx==1):

                valid_prefetcher.reset()
                val_batch_data = valid_prefetcher.next()

                synmodule.eval()

                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results'), exist_ok=True)
                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'rec_results'), exist_ok=True)

                while val_batch_data is not None:
                    image_lq = val_batch_data["lq"].cuda()
                    image_gt = val_batch_data["gt"].cuda()
                    image_basename =  str(val_batch_data["name"][0])

                    if (image_basename in savelist):
                        with torch.no_grad():
                            output = resmodule(image_lq)
                            output = synmodule(output)
                            rec_image = postprocess(output.clone())[0]
                            cv2.imwrite(os.path.join(work_dir, job_name, f'{batch_idx}_results','rec_results', image_basename+'.png'), rec_image)
                    
                    val_batch_data = valid_prefetcher.next()

                if batch_idx % 4000 == 0:
                    ckpt = {"fencoder": synmodule.fencoder.state_dict(),
                            "wmodule": synmodule.wmodule.state_dict(),
                            "generator": synmodule.generator.state_dict(),
                            "discriminator": discriminator.state_dict()}

                    optim = {
                        "optimizer": optimizer.state_dict(),
                        "optimizer_D": optimizer_D.state_dict(),
                    }

                    torch.save(ckpt, os.path.join(work_dir, job_name, f'synmodule_{batch_idx}.pth'))
                    torch.save(optim, os.path.join(work_dir, job_name, f'optim_latest.pth'))
        
                if batch_idx ==args.final_batch_idx:
                    break

        if batch_idx ==args.final_batch_idx:
            break

    ckpt = {"fencoder": synmodule.fencoder.state_dict(),
            "wmodule": synmodule.wmodule.state_dict(),
            "generator": synmodule.generator.state_dict(),
            "discriminator": discriminator.state_dict()}

    optim = {
        "optimizer": optimizer.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
    }
    torch.save(ckpt, os.path.join(work_dir, job_name, f'synmodule_latest.pth'))
    torch.save(optim, os.path.join(work_dir, job_name, f'optim_latest.pth'))                   


def load_dataset(args) -> [CUDAPrefetcher, CUDAPrefetcher]:

    train_datasets = PairedImageDataset(args.train_lq_dir, args.train_gt_dir)
    valid_datasets = PairedImageDataset(args.test_lq_dir, args.test_gt_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=args.batch_num,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    
    device = torch.device("cuda", 0)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, device)

    return train_prefetcher, valid_prefetcher

if __name__ == '__main__':
    main()
