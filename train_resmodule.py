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
from dataset import *
import random
from models.resmodule import ResModule
from loss.PSNRloss import PSNRLoss
from utils import calculate_psnr

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--job_name', type=str, default='nafnet_deblur')
    parser.add_argument('--save_dir', type=str, default='./training/ResModule')
    
    # Dataset path
    parser.add_argument('--train_gt_dir', type=str, default='')
    parser.add_argument('--train_lq_dir', type=str, default='')
    parser.add_argument('--test_gt_dir', type=str, default='')
    parser.add_argument('--test_lq_dir', type=str, default='')

    #training options
    parser.add_argument('--batch_num', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--initial_lr', type=float, default=1e-4)
    parser.add_argument('--regression_path', type=str, default='./ckpt/pretrained/nafnet_deblur.pth')
    parser.add_argument('--type', type=str, default='nafnet_deblur')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--final_batch_idx', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=777)

    # if use LSUN-church dataset
    parser.add_argument('--church', action='store_true')

    # if use l1 loss instead of psnr loss
    parser.add_argument('--l1', action='store_true')

    return parser.parse_args()

def main():
    wandb.init(project="train_resmodule")
    args = parse_args()

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

    print(args.type)

    # define restoration module, optimizer, loss, dataset
    
    resmodule = ResModule(args)
    resmodule.cuda().requires_grad_(True).train()
    resmodule.load_reg()

    optimizer = torch.optim.Adam(resmodule.parameters(), lr=args.initial_lr)

    cal_l1_loss = nn.L1Loss()
    cal_psnr_loss = PSNRLoss()

    train_prefetcher, valid_prefetcher = load_dataset(args)

    savelist=['3030','3077','3000','3001','3004','3006','3012','3022','3031','3048','3072']
    
    best_psnr=-1.
    best_batch_idx=0
    batch_idx = 0

    while (True):
        pbar = tqdm(total=len(train_prefetcher))
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        # Training Resmodule
        while batch_data is not None:

            resmodule.train()

            lq = batch_data['lq'].cuda()
            gt = batch_data['gt'].cuda()

            I_output = resmodule(lq)

            if not args.l1:
                total_loss = cal_psnr_loss(I_output, gt)
            else:
                total_loss = cal_l1_loss(I_output, gt)

            if batch_idx % 50==0:
                wandb.log({"loss":total_loss.item()})

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_data = train_prefetcher.next()

            batch_idx += 1
            pbar.update(1)

            # Evalutaion part
            if (batch_idx % args.save_interval == 0):

                resmodule.eval()
                valid_prefetcher.reset()

                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results'), exist_ok=True)
                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'rec_results'), exist_ok=True)
                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'ground_truth'), exist_ok=True)
                os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'degradation'), exist_ok=True)            

                val_batch_data = valid_prefetcher.next()
                psnr = 0
                
                while val_batch_data is not None:
                    image_lq = val_batch_data["lq"].cuda()
                    image_gt = val_batch_data["gt"].cuda()
                    image_basename =  str(val_batch_data["name"][0])

                    with torch.no_grad():
                        I_output = resmodule(image_lq)
                        rec_image = postprocess(I_output.clone())[0]
                        image_gt = postprocess(image_gt.clone())[0]

                        if (image_basename in savelist):
                            # save some images
                            cv2.imwrite(os.path.join(work_dir, job_name, f'{batch_idx}_results','ground_truth', image_basename+'.png'),image_gt)
                            cv2.imwrite(os.path.join(work_dir, job_name, f'{batch_idx}_results','degradation', image_basename+'.png'), postprocess(image_lq.clone())[0])
                            cv2.imwrite(os.path.join(work_dir, job_name, f'{batch_idx}_results','rec_results', image_basename+'.png'), rec_image)

                        psnr+=calculate_psnr(rec_image, image_gt)
                        val_batch_data = valid_prefetcher.next()

                psnr /= len(valid_prefetcher)
                print(f"batch idx: {batch_idx}, psnr: {psnr}")
                wandb.log({"PSNR": psnr})

                ckpt = resmodule.state_dict()
                optim = optimizer.state_dict()

                if psnr>=best_psnr:
                    best_psnr = psnr
                    best_batch_idx = batch_idx
                    torch.save(ckpt, os.path.join(work_dir, job_name, f'resmodule_best_{batch_idx}.pth'))

                print(f"=====   best model:    iter {best_batch_idx}   psnr {best_psnr} =========")
                torch.save(ckpt, os.path.join(work_dir, job_name, f'resmodule_latest.pth'))
                torch.save(optim, os.path.join(work_dir, job_name, f'optim_latest.pth'))

            if batch_idx==args.final_batch_idx:
                break
        
        if batch_idx == args.final_batch_idx:
            break

    print(f"============Training Finished============")
    print(f"=====   best model:    iter {best_batch_idx}   psnr {best_psnr} =========")
    ckpt = resmodule.state_dict()
    optim = optimizer.state_dict()
    torch.save(ckpt, os.path.join(work_dir, job_name, f'resmodule_latest.pth'))
    torch.save(optim, os.path.join(work_dir, job_name, f'optim_latest.pth'))                   


def load_dataset(args) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
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
