# Super Resolution

## Methods

### Bibubic

| Dataset  | Scale              | PSNR                        | SSIM                           | IFC                      |
| -------- | ------------------ | --------------------------- | ------------------------------ | ------------------------ |
| Set5     | x2<br />x3<br />x4 | 33.64<br />30.39<br />28.42 | 0.9292<br />0.8678<br />0.8101 | 5.72<br />3.45<br />2.28 |
| Set14    | x2<br />x3<br />x4 | 30.22<br />27.53<br />25.99 | 0.8683<br />0.7737<br />0.7023 | 5.74<br />3.33<br />2.18 |
| BSD100   | x2<br />x3<br />x4 | 29.55<br />27.20<br />25.96 | 0.8425<br />0.7382<br />0.6672 |                          |
| Urban100 | x2<br />x3<br />x4 | 26.66<br />24.64<br />23.14 | 0.8408<br />0.7353<br />0.6573 | 5.72<br />-<br />2.27    |

### SRCNN(2014)

paper:

- [Learning a Deep Convolutional Network for Image Super-Resolution(ECCV 2014)](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf)

- [Image Super-Resolution Using Deep Convolutional Networks (TPIAMI 2015)](https://arxiv.org/abs/1501.00092)

Dataset prepare: 91-image, 33x33, stride:14

Todo: 91-image, train: 33x33, test: 21x21, for 9-1-5; for train 33x33, test: 17x17, for 9-5-5), see source code on http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html.

The best result of the paper is trained on 395,909 images from the ILSVRC 2013 ImageNet detection training partition.

I simply pick the first 1000 images from the validation dataset of ILSVRC 2012.

| Dataset   | Scale              | PSNR(91-images/ImageNet/paper)                               | SSIM(91-images/ImageNet/paper) |
| --------- | ------------------ | ------------------------------------------------------------ | ------------------------------ |
| Set5      | x2<br />x3<br />x4 | 36.55/36.64/36.66/<br />32.18/32.45/32.75<br />30.34/30.36/30.49 |                                |
| Set14     | x2<br />x3<br />x4 | 32.33/32.35/32.45<br />28.84/29.06/29.28<br />27.37/27.41/27.49 |                                |
| BSD100    | x2<br />x3<br />x4 | 31.32/31.30/31.36<br />28.24/28.25/28.41<br />26.81/26.85/26.90 |                                |
| Urban 100 | x2<br />x3<br />x4 | 29.05/29.00/29.50<br />25.38/25.31/26.24<br />24.33/24.36/24.52 |                                |

### FSRCNN(2016)

paper: [Accelerating the Super-Resolution Convolutional Neural Network（ECCV）](https://arxiv.org/abs/1608.00367)

Dataset prepare: 91-image

Fine tune: General-100 (actually without this to get the following answer :)

For 2x, HR: 20x20, stride:2, LR: 10x10

For 3x, HR: 21x21, stride:3, LR: 7x7

For 4x, HR: 24x24, stride:4, LR: 6x6

Scale: 1.0 0.9 0.8 0.7 0.6, rotation: 0 90 180 270

It takes a very long time to train.

| Dataset  | Scale              | PSNR(91-images/paper)                         | SSIM(91-images/paper) |
| -------- | ------------------ | --------------------------------------------- | --------------------- |
| Set5     | x2<br />x3<br />x4 | 36.94/36.66<br />33.04/32.75<br />30.66/30.48 |                       |
| Set14    | x2<br />x3<br />x4 | 32.48/34.54<br />29.33/29.28<br />27.43/27.49 |                       |
| BSD100   | x2<br />x3<br />x4 | 31.42/31.73<br />28.47/28.55<br />26.93/26.92 |                       |
| Urban100 | x2<br />x3<br />x4 | 29.30/29.81<br />26.61/-<br />24.50/24.61     |                       |

### VDSR(2016)

paper: [Accurate Image Super-Resolution Using Very Deep Convolutional Networks（CVPR）](http://arxiv.org/abs/1511.04587)

Dataset prepare:  91-image, Bsd300 train set, 41x41, stride:41, scale: 1.0 0.7 0.5, rotation: 0 90 180 270, flip: 0 1 2, upscaleFactor: 2 3 4, single model

| Dataset  | Scale              | PSNR(291-images/paper)                        | SSIM(291-images/paper)                              |
| -------- | ------------------ | --------------------------------------------- | --------------------------------------------------- |
| Set5     | x2<br />x3<br />x4 | 37.46/37.53<br />33.67/33.66<br />31.33/31.35 | 0.9574/0.9587<br />0.9211/0.9213<br />0.8827/0.8838 |
| Set14    | x2<br />x3<br />x4 | 32.84/33.03<br />29.76/29.77<br />27.98/28.01 | 0.9110/0.9124<br />0.8314/0.8314<br />0.7669/0.7674 |
| BSD100   | x2<br />x3<br />x4 | 31.81/31.90<br />28.81/28.82<br />27.26/26.90 | 0.8942/0.8960<br />0.7970/0.7976<br />0.7242/0.7251 |
| Urban100 | x2<br />x3<br />x4 | 30.18/30.76<br />26.09/27.14<br />25.15/25.18 | 0.9165/0.9140<br />0.8358/0.8279<br />0.7504/0.7524 |

### ESPCN(2017)

paper: [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network（CVPR)](https://arxiv.org/abs/1609.05158)

Dataset prepare: 91-image, ImageNet 50,000 randomly selected images.

for 2x, HR: 34x34, stride: 34, LR: 17x17

for 3x, HR: 51x51, stride: 51, LR: 17x17

for 4x, HR: 68x68, stride: 68, LR: 17x17

| Dataset | Scale              | PSNR(91-images/paper/ImageNet/paper)                         | SSIM(91-images/paper/ImageNet/paper) |
| ------- | ------------------ | ------------------------------------------------------------ | ------------------------------------ |
| Set5    | x2<br />x3<br />x4 | 36.35/-/35.64/-<br />32.32/32.75/31.95/33.00<br />30.66/-/30.14/30.90 |                                      |
| Set14   | x2<br />x3<br />x4 | 32.13/-/31.79<br />28.94/29.28/28.76/29.42<br />27.43/-/27.23/27.73 |                                      |
| BSD200  | x2<br />x3<br />x4 | 31.08/-/30.88/-<br />28.16/28.55/28.04/28.52<br />26.93/-/26.77/27.06 |                                      |

### DRCN(2016)

paper: [Deeply-Recursive Convolutional Network for Image Super-Resolution（CVPR）](https://arxiv.org/abs/1511.04491)

Dataset prepare: 91-image, 41x41, stride: 21, flip: 0 1, rotation: 0 90 180 270, uf: 2/3/4, single model

scales 1.0 0.7 0.5

| Dataset  | Scale              | PSNR(91-images/paper)                 | SSIM(91-images/paper)                     |
| -------- | ------------------ | ------------------------------------- | ----------------------------------------- |
| Set5     | x2<br />x3<br />x4 | 34.99/37.63<br />-/33.82<br />-/31.53 | 0.9420/0.9588<br />-/0.9226<br />-/0.8854 |
| Set14    | x2<br />x3<br />x4 | 31.28/33.04<br />-/29.76<br />-/28.02 | 0.8918/0.9118<br />-/0.8311<br />-/0.7670 |
| BSD100   | x2<br />x3<br />x4 | 30.42/31.85<br />-/28.80<br />-/27.23 | 0.8709/0.8942<br />-/0.7963<br />-/0.7233 |
| Urban100 | x2<br />x3<br />x4 | -/30.75<br />-/27.15<br />-/25.14     | 0/0.9133<br />-/0.8276<br />-/0.7510      |

### DRRN(2017)

paper:  [Image super-resolution via deep recursive residual network（CVPR）](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf)

Dataset prepare: 91-image, BSD300 train set, 31x31, stride: 21, rotation: 0 90 180 270, flip: 0 1, uf: 2 3 4, single model

| Dataset  | Scale              | PSNR(291-images/paper)                        | SSIM(91-images/paper)                               |
| -------- | ------------------ | --------------------------------------------- | --------------------------------------------------- |
| Set5     | x2<br />x3<br />x4 | 37.52/37.74<br />33.74/34.03<br />31.34/31.68 | 0.9577/0.9591<br />0.9219/0.9244<br />0.8836/0.8888 |
| Set14    | x2<br />x3<br />x4 | 32.87/33.23<br />29.77/29.96<br />28.01/28.21 | 0.9109/0.9136<br />0.8314/0.8349<br />0.7672/0.7720 |
| BSD100   | x2<br />x3<br />x4 | 31.78/32.05<br />28.79/28.95<br />27.25/27.38 | 0.8940/0.8973<br />0.7966/0.8004<br />0.7241/0.7284 |
| Urban100 | x2<br />x3<br />x4 | 30.04/31.23<br />27.45/27.53<br />25.15/25.44 | 0.9168/0.9188<br />0.8382/0.8378<br />0.7522/0.7638 |

### LapSRN(2016)



### EDSR/MDSR(NTIRE 2017)



### SRGAN/SRResNet(2016)



### RCAN

### WDSR



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --network=pretrained_gans/Gs.pth --model checkpoints/39_PL_netG.pth --dis_model checkpoints/39_PL_netD.pth --resume True --start_iter 40

