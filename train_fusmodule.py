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
from data_augmentation import RandomAugmentation
from dataset import *
import random
from models.resmodule import ResModule
from models.synmodule import SynModule
from models.fusmodule import FusionNet
from loss.contextual import PatchContextualLoss
from loss.Lpips import LpipsLoss
from archs.RRDBNet import RRDBNetX8
from utils import yaml_load, calculate_psnr

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--job_name', type=str, default='nafnet_deblur')
    parser.add_argument('--save_dir', type=str, default='./training/FusModule')


    # Dataset path
    parser.add_argument('--train_gt_dir', type=str, default='')
    parser.add_argument('--train_lq_dir', type=str, default='')
    parser.add_argument('--test_gt_dir', type=str, default='')
    parser.add_argument('--test_lq_dir', type=str, default='')

    # Module path
    parser.add_argument('--resmodule_path', type=str, default='./ckpt/UGPNet_with_NAFNet_deblur/resmodule_best_20000.pth')
    parser.add_argument('--synmodule_path', type=str, default='./ckpt/UGPNet_with_NAFNet_deblur/synmodule_40000.pth')

    #training options
    parser.add_argument('--batch_num', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--perceptual_weight', type=float, default=10)
    parser.add_argument('--context_weight', type=float, default=0.05)
    parser.add_argument('--type', type=str, default="nafnet_deblur")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--spatial_size', type=int, default=16)
    parser.add_argument('--milestone', type= int, default=8000)
    parser.add_argument('--vgg_layer', type=str, default='relu3_4')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--final_batch_idx', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=777)
    
    # if use LSUN-church dataset
    parser.add_argument('--church', action="store_true")


    return parser.parse_args()

def main():

    args = parse_args()
    args.randomize_noise=False
    wandb.init(project="train_fusion")

    wandb.run.name = f'{args.job_name}'
    wandb.run.save()
    wandb.config.update(args)

    # Set random seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    work_dir = args.save_dir
    job_name = args.job_name
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)

    # Save current file and arguments
    current_file_path = inspect.getfile(inspect.currentframe())
    current_file_name = os.path.basename(current_file_path)
    shutil.copyfile(current_file_path, os.path.join(work_dir, job_name, current_file_name))
    with open(os.path.join(work_dir, job_name, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    chan = 128 if args.church else 64
    fusmodule = FusionNet(chan=chan)
    fusmodule.cuda().requires_grad_(True).train()

    #resmodule
    if "rrdb" in args.type:
        # Direct approach dont use additional networks
        resmodule = RRDBNetX8(3, 3)
        resmodule.cuda().requires_grad_(False).eval()
        resmodule.load_state_dict(torch.load(args.resmodule_path)['params_ema'])
    else:
        # residual approach use additional networks
        resmodule = ResModule(args)
        resmodule.cuda().requires_grad_(False).eval()
        resmodule.load_state_dict(torch.load(args.resmodule_path))     

    #synmodule
    synmodule = SynModule(args)
    synmodule.cuda().requires_grad_(False).eval()
    synmodule.load()


    #optimzier and scheduler
    optimizer = torch.optim.Adam(fusmodule.parameters(), lr=args.initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.milestone], gamma=0.1)


    #losses
    cal_perceptual_loss = LpipsLoss()
    cal_contextual_loss = PatchContextualLoss(vgg_layer=args.vgg_layer,patch_size=args.patch_size).cuda()
    cal_l1_loss = nn.L1Loss()

    # datasets
    train_prefetcher, valid_prefetcher = load_dataset(args)

    savelist=['3030','3077','3000','3001','3004','3006','3012','3022','3031','3048','3072']

    best_psnr=-1.
    best_batch_idx=0
    batch_idx = 0

    while(True):
        pbar = tqdm(total=len(train_prefetcher))
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:

            fusmodule.train()

            lq = batch_data['lq'].cuda()
            gt = batch_data['gt'].cuda()
            
            with torch.no_grad():
                I_R, F_R = resmodule(lq, return_f=True)
                I_G, F_G = synmodule(I_R, return_f=True)

            I_F = fusmodule(F_G, F_R)

            l1_loss = cal_l1_loss(gt, I_F)
            per_loss = cal_perceptual_loss(gt, I_F)
            context_loss = cal_contextual_loss(I_F, I_G)

            total_loss = l1_loss + args.perceptual_weight*per_loss + args.context_weight*context_loss


            loss_dict = {
                "l1 loss":l1_loss.item(),
                "lpips loss":per_loss.item(),
                "context loss":context_loss.item()
                }


            if batch_idx % 50==0:
                wandb.log(loss_dict)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            batch_data = train_prefetcher.next()

            batch_idx += 1
            pbar.update(1) 

            # Evalutaion part
            if (batch_idx % args.save_interval == 0) or batch_idx==1:
                
                valid_prefetcher.reset()
                val_batch_data = valid_prefetcher.next()
                image_num = len(valid_prefetcher)
                fusmodule.eval()
                psnr = 0

                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results'), exist_ok=True)
                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'rec_results'), exist_ok=True)


                while val_batch_data is not None:
                    image_lq = val_batch_data["lq"].cuda()
                    image_gt = val_batch_data["gt"].cuda()
                    image_basename =  str(val_batch_data["name"][0])

                    with torch.no_grad():

                        I_R, F_R = resmodule(image_lq, return_f=True)
                        I_G, F_G = synmodule(I_R, return_f=True)
                        output = fusmodule(F_G, F_R)

                        rec_image = postprocess(output.clone())[0]
                        image_gt = postprocess(image_gt.clone())[0]
                        psnr+=calculate_psnr(rec_image, image_gt)

                        if (image_basename in savelist):
                            # save some images
                            cv2.imwrite(os.path.join(work_dir, job_name, f'{batch_idx}_results','rec_results', image_basename+'.png'), rec_image)
                            
                    val_batch_data = valid_prefetcher.next()

                psnr /= image_num
                print(f"batch idx: {batch_idx}, psnr: {psnr}")
                wandb.log({"PSNR": psnr})
                if psnr>=best_psnr:
                    ckpt = fusmodule.state_dict()
                    optim = optimizer.state_dict()
                    best_psnr = psnr
                    best_batch_idx = batch_idx
                    torch.save(fusmodule.state_dict(), os.path.join(work_dir, job_name, f'fusmodule_best.pth'))

                print(f"=====   best psnr model:    iter {best_batch_idx}   psnr {best_psnr} =========")


                ckpt = fusmodule.state_dict()
                optim = optimizer.state_dict()
                torch.save(ckpt, os.path.join(work_dir, job_name, f'fusmodule_{batch_idx}.pth'))
                torch.save(optim, os.path.join(work_dir, job_name, f'optim_latest.pth'))

            if args.final_batch_idx == batch_idx:
                break
        
        if args.final_batch_idx == batch_idx:
            break

    print(f"=====   best psnr model:    iter {best_batch_idx}   psnr {best_psnr} =========")

    ckpt = fusmodule.state_dict()
    optim = optimizer.state_dict()
    torch.save(ckpt, os.path.join(work_dir, job_name, f'fusmodule_latest.pth'))
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
