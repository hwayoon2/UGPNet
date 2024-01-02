from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
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
from dataset import *
from models.UGPNet import UGPNet

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--job_name', type=str, default='nafnet_denoise')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--type', type=str, default='nafnet_denoise')

    parser.add_argument('--resmodule_path', type=str, default='./ckpt/UGPNet_with_NAFNet_denoise/resmodule_best_18000.pth')
    parser.add_argument('--synmodule_path', type=str, default='./ckpt/UGPNet_with_NAFNet_denoise/synmodule_40000.pth')
    parser.add_argument('--fusmodule_path', type=str, default='./ckpt/UGPNet_with_NAFNet_denoise/fusmodule_best.pth')

    parser.add_argument('--spatial_size', type=int, default=16)
    parser.add_argument('--randomize_noise', action="store_true")
    parser.add_argument('--church', action="store_true")

    parser.add_argument('--save_all', action='store_true')

    parser.add_argument('--test_lq_dir', type=str, default="./sample_images/noisy")
    parser.add_argument('--test_gt_dir', type=str, default="./sample_images/gt")

    return parser.parse_args()

def main():

    args = parse_args()

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

    ugpnet = UGPNet(args)
    ugpnet.cuda().requires_grad_(False).eval()
    ugpnet.load_ckpt()
    
    valid_prefetcher = load_dataset(args)
    valid_prefetcher.reset()
    val_batch_data = valid_prefetcher.next()
    image_num = len(valid_prefetcher)

    os.makedirs(os.path.join(work_dir, job_name, f'results'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name, f'results', 'rec_results'), exist_ok=True)
    if args.save_all:
        os.makedirs(os.path.join(work_dir, job_name, f'results', 'resmodule_results'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, job_name, f'results', 'synmodule_results'), exist_ok=True)

    while val_batch_data is not None:
        image_lq = val_batch_data["lq"].cuda()
        image_basename =  str(val_batch_data["name"][0])

        with torch.no_grad():

            if args.save_all:
                x_res, x_syn , x_fus = ugpnet(image_lq, return_all=True)
                rec_image = postprocess(x_syn.clone())[0]
                cv2.imwrite(os.path.join(work_dir, job_name, f'results','synmodule_results', image_basename+'.png'), rec_image)
                rec_image = postprocess(x_res.clone())[0]
                cv2.imwrite(os.path.join(work_dir, job_name, f'results','resmodule_results', image_basename+'.png'), rec_image)
                rec_image = postprocess(x_fus.clone())[0]
                cv2.imwrite(os.path.join(work_dir, job_name, f'results','rec_results', image_basename+'.png'), rec_image)
                    
            else:
                x_fus = ugpnet(image_lq)
                rec_image = postprocess(x_fus.clone())[0]
                cv2.imwrite(os.path.join(work_dir, job_name, f'results','rec_results', image_basename+'.png'), rec_image)

        val_batch_data = valid_prefetcher.next()


def load_dataset(args) -> [CUDAPrefetcher]:
    valid_datasets = PairedImageDataset(args.test_lq_dir, args.test_gt_dir)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    device = torch.device("cuda", 0)

    # Place all data on the preprocessing data loader
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, device)
    return valid_prefetcher

if __name__ == '__main__':
    main()
