## UGPNet: Universal Generative Prior for Image Restoration<br><sub>Official PyTorch Implementation of the WACV 2024 Paper</sub>

**UGPNet: Universal Generative Prior for Image Restoration**<br>
Hwayoon Lee, Kyoungkook Kang, Hyeongmin Lee, Seung-Hwan Baek, Sunghyun Cho<br>

[\[Paper\]](https://arxiv.org/abs/2401.00370)


### Environment Setting
    conda env create -f environment.yaml
    conda activate ugpnet
    export BASICSR_JIT=True # We use basicsr library

### Training
UGPNet consists of three modules, and each module is trained sequentially. When you execute the following commands, you can train UGPNet with NAFNet (deblur) using the provided pretrained weights. You can modify the arguments to train with your own dataset, model, and degradation. Please refer to the training codes.

#### 1. Restoration module

        python train_resmodule.py

#### 2. Synthesis module
        
        python train_synmodule.py

#### 3. Fusion Module

        python train_fusmodule.py

### Inference
 We provide pretrained weights of UGPNet with NAFNet (denoising, deblurring) and UGPNet with RRDBNet (super-resolution) and some sample images.
 [[Download]](https://drive.google.com/drive/folders/1fTwlaFFnuvzev371jhCxL1fFlIEl7wA-?usp=drive_link)
 
* Image Denoising

        test_scripts/test_UGPNet_w_NAFNet_denoise.sh

* Image Deblurring
    
        test_scripts/test_UGPNet_w_NAFNet_deblur.sh

* Super-resolution X8


        test_scripts/test_UGPNet_w_RRDBNet_sr.sh