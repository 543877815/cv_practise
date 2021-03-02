# Super Resolution

## Methods

### Bibubic

| Dataset  | Scale              | PSNR                        | SSIM                           | IFC                      |
| -------- | ------------------ | --------------------------- | ------------------------------ | ------------------------ |
| Set5     | x2<br />x3<br />x4 | 33.64<br />30.39<br />28.42 | 0.9292<br />0.8678<br />0.8101 | 5.72<br />3.45<br />2.28 |
| Set14    | x2<br />x3<br />x4 | 30.22<br />27.53<br />25.99 | 0.8683<br />0.7737<br />0.7023 | 5.74<br />3.33<br />2.18 |
| BSD100   | x2<br />x3<br />x4 | 29.55<br />27.20<br />25.96 |                                |                          |
| Urban100 | x2<br />x3<br />x4 | 26.66<br />24.64<br />23.14 | 0.8408<br />-<br />0.6573      | 5.72<br />-<br />2.27    |

### SRCNN

paper:

- [Learning a Deep Convolutional Network for Image Super-Resolution(ECCV 2014)](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf)

- [Image Super-Resolution Using Deep Convolutional Networks (TPIAMI 2015)](https://arxiv.org/abs/1501.00092)

Dataset prepare: 91-image, 33x33, stride:14

The best result of the paper is trained on 395,909 images from the ILSVRC 2013 ImageNet detection training partition.

I simply pick the first 1000 images from the validation dataset of ILSVRC 2012.

| Dataset   | Scale              | PSNR(91-images/ImageNet/paper)                               | SSIM(91-images/ImageNet/paper) |
| --------- | ------------------ | ------------------------------------------------------------ | ------------------------------ |
| Set5      | x2<br />x3<br />x4 | 36.55/36.64/36.66/<br />32.18/32.45/32.75<br />30.34/30.36/30.49 |                                |
| Set14     | x2<br />x3<br />x4 | 32.33/32.35/32.45<br />28.84/29.06/29.28<br />27.37/27.41/27.49 |                                |
| BSD100    | x2<br />x3<br />x4 | 31.32/31.30/31.36<br />28.24/28.25/28.41<br />26.81/26.85/26.90 |                                |
| Urban 100 | x2<br />x3<br />x4 | 29.05/29.00/29.50<br />25.38/25.31/26.24<br />24.33/24.36/24.52 |                                |

### FSRCNN

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

### VDSR

paper: [Accurate Image Super-Resolution Using Very Deep Convolutional Networks（CVPR）](http://arxiv.org/abs/1511.04587)

Dataset prepare:  91-image, Bsd300 train set, 41x41, stride:41, scale: 1.0 0.7 0.5, rotation: 0 90 180 270, flip: 0 1 2, uf: 2 3 4, single model

| Dataset  | Scale              | PSNR(91-images/paper)                         | SSIM(91-images/paper) |
| -------- | ------------------ | --------------------------------------------- | --------------------- |
| Set5     | x2<br />x3<br />x4 | 37.46/37.53<br />33.67/33.66<br />31.33/31.35 |                       |
| Set14    | x2<br />x3<br />x4 | 32.84/33.03<br />29.76/29.77<br />27.98/28.01 |                       |
| BSD100   | x2<br />x3<br />x4 | 31.81/31.90<br />28.81/28.82<br />27.26/26.90 |                       |
| Urban100 | x2<br />x3<br />x4 | 30.18/30.76<br />26.09/27.14<br />25.15/25.18 |                       |

### ESPCN

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

### DRCN

paper: [Deeply-Recursive Convolutional Network for Image Super-Resolution（CVPR）](https://arxiv.org/abs/1511.04491)

Dataset prepare: 91-image, 41x41, stride: 21, flip: 0 1 2 3, uf: 2 3 4, single model

### DRRN

paper:  [Image super-resolution via deep recursive residual network（CVPR）](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf)

Dataset prepare: 91-image, BSD300 train set, 31x31, stride: 21, rotation: 0 90 180 270, flip: 0 1, uf: 2 3 4, single model

| Dataset  | Scale              | PSNR(291-images/paper)                | SSIM(91-images/paper) |
| -------- | ------------------ | ------------------------------------- | --------------------- |
| Set5     | x2<br />x3<br />x4 | 37.30/37.74<br />-/34.03<br />-/31.68 |                       |
| Set14    | x2<br />x3<br />x4 | 32.71/33.23<br />-/29.96<br />-/28.21 |                       |
| BSD100   | x2<br />x3<br />x4 | 31.65/32.05<br />-/28.95<br />-/27.38 |                       |
| Urban100 | x2<br />x3<br />x4 | 29.75/31.23<br />-/27.53<br />-/25.44 |                       |

## Results

The method transforms RGB to YCrCb is mainly refer to matlab' s `rgb2ycrcb` function. The PSNR is calculated only on the y channel on the [YCrCb](https://en.wikipedia.org/wiki/YCbCr) color space, and the detail Implementation is refer to the code in [SelfExSR](https://github.com/jbhuang0604/SelfExSR/blob/master/quant_eval/compute_difference.m), which crops some margin pixels during testing in order to get the similar result as the paper.





Results on Set 5

| Scale       | BicuBic | SRCNN        | FSRCNN(not finish) | VDSR         | ESPCN/91/ImageNet | DRCN | DRRN |
| ----------- | ------- | ------------ | ------------------ | ------------ | ----------------- | ---- | ---- |
| **2x**-PSNR | 33.64   | 35.84(36.66) | 36.81(37.00)       | 37.15(37.53) | 36.36()           |      |      |
| **3x**-PSNR | 30.39   | 32.18(32.75) | 32.83(33.16)       | 33.30(33.66) | 32.32(32.55)      |      |      |
| **4x**-PSNR | 28.42   | 29.94(30.48) | 28.71(30.71)       | 30.55(31.35) | 29.94()           |      |      |
| **2x**-SSIM | 0.9292  |              |                    |              |                   |      |      |
| **3x**-SSIM | 0.8678  |              |                    |              |                   |      |      |
| **4x**-SSIM | 0.8101  |              |                    |              |                   |      |      |
| **2x**-IFC  | 5.72    |              |                    |              |                   |      |      |
| **3x**-IFC  | 3.45    |              |                    |              |                   |      |      |
| **4x**-IFC  | 2.28    |              |                    |              |                   |      |      |

Results on Set 14

| Scale       | BicuBic | SRCNN        | FSRCNN       | VDSR         | ESPCN/91/ImageNet |
| ----------- | ------- | ------------ | ------------ | ------------ | ----------------- |
| **2x**-PSNR | 30.22   | 31.81(34.42) | 32.35(32.63) | 32.64(33.03) | 32.13             |
| **3x**-PSNR | 27.53   | 28.84(29.28) | 29.23(29.43) | 29.53(29.77) | 28.94(29.08)      |
| **4x**-PSNR | 25.99   | 27.04(27.49) | 27.42(27.59) | 27.89(28.01) | 27.05()           |
| **2x**-SSIM | 0.8683  |              |              |              |                   |
| **3x**-SSIM | 0.7737  |              |              |              |                   |
| **4x**-SSIM | 0.7023  |              |              |              |                   |
| **2x**-IFC  | 5.74    |              |              |              |                   |
| **3x**-IFC  | 3.33    |              |              |              |                   |
| **4x**-IFC  | 2.18    |              |              |              |                   |

Results on Urban 100

| Scale       | BicuBic | SRCNN | FSRCNN | VDSR |
| ----------- | ------- | ----- | ------ | ---- |
| **2x**-PSNR | 26.66   |       |        |      |
| **3x**-PSNR | 23.98   |       |        |      |
| **4x**-PSNR | 23.14   |       |        |      |
| **2x**-SSIM | 0.8408  |       |        |      |
| **3x**-SSIM |         |       |        |      |
| **4x**-SSIM | 0.6573  |       |        |      |
| **2x**-IFC  | 5.72    |       |        |      |
| **3x**-IFC  |         |       |        |      |
| **4x**-IFC  | 2.27    |       |        |      |

Results on BSD 100 

| Scale       | BicuBic | SRCNN | FSRCNN | VDSR |
| ----------- | ------- | ----- | ------ | ---- |
| **2x**-PSNR | 29.55   |       |        |      |
| **3x**-PSNR | 27.20   | 28.57 |        |      |
| **4x**-PSNR | 25.96   |       |        |      |
| **2x**-SSIM | 0.8425  |       |        |      |
| **3x**-SSIM | 0.7382  |       |        |      |
| **4x**-SSIM | 0.6672  |       |        |      |
| **2x**-IFC  | 5.26    |       |        |      |
| **3x**-IFC  | 3.00    |       |        |      |
| **4x**-IFC  | 1.91    |       |        |      |