# Super Resolution

## Methods

### Bibubic

| Dataset  | Scale              | PSNR                        | SSIM                           | IFC                      |
| -------- | ------------------ | --------------------------- | ------------------------------ | ------------------------ |
| Set5     | x2<br />x3<br />x4 | 33.64<br />30.39<br />28.42 | 0.9292<br />0.8678<br />0.8101 | 5.72<br />3.45<br />2.28 |
| Set14    | x2<br />x3<br />x4 | 30.22<br />27.53<br />25.99 | 0.8683<br />0.7737<br />0.7023 | 5.74<br />3.33<br />2.18 |
| BSD100   | x2<br />x3<br />x4 | 27.20                       |                                |                          |
| Urban100 | x2<br />x3<br />x4 | 26.66<br />-<br />23.14     | 0.8408<br />-<br />0.6573      | 5.72<br />-<br />2.27    |

### SRCNN

Dataset prepare: 91-image, 33x33, stride:14

The best result of the paper is trained on ImageNet.

| Dataset | Scale              | PSNR(91-images/ImageNet/paper)                       | SSIM(91-images/ImageNet/paper) |
| ------- | ------------------ | ---------------------------------------------------- | ------------------------------ |
| Set5    | x2<br />x3<br />x4 | 35.84/-/36.66/<br />32.18/-/32.75<br />29.94/-/30.48 |                                |
| Set14   | x2<br />x3<br />x4 | 31.82/-/34.42<br />28.84/-/29.28<br />27.04/-/27.49  |                                |
| BSD200  | x2<br />x3<br />x4 | -/-/30.29<br />-/-/27.18<br />-/-/25.60              |                                |

### FSRCNN

Dataset prepare: 91-image

Fine tune: General-100(并没有做)

For 2x, HR: 20x20, stride:2, LR: 10x10

For 3x, HR: 21x21, stride:3, LR: 7x7

For 4x, HR: 24x24, stride:4, LR: 6x6

Scale: 1.0 0.9 0.8 0.7 0.6, rotation: 0 90 180 270

It takes a very long time to train.

| Dataset | Scale              | PSNR(91-images/paper)                         | SSIM(91-images/paper) |
| ------- | ------------------ | --------------------------------------------- | --------------------- |
| Set5    | x2<br />x3<br />x4 | 36.94/36.66<br />33.04/32.75<br />30.66/30.48 |                       |
| Set14   | x2<br />x3<br />x4 | 32.49/34.54<br />29.33/29.28<br />27.43/27.49 |                       |
| BSD100  | x2<br />x3<br />x4 | 31.42/31.73<br />28.47/28.55<br />26.93/26.92 |                       |

### VDSR

Dataset prepare: 91-image, Bsd200 train set, 41x41, stride:41, scale: 1.0 0.9 0.8 0.7 0.6, rotation: 0 90 180 270

Todo: 91-image, Bsd300 train set, 41x41, stride:41, scale: 1.0 0.7 0.5, rotation: 0 90 180 270, flip: 0 1 2, uf 2 3 4, single model

### ESPCN

Dataset prepare: 91-image, ImageNet 50,000 randomly selected images.

for 2x, HR: 34x34, stride: 34, LR: 17x17

for 3x, HR: 51x51, stride: 51, LR: 17x17

for 4x, HR: 68x68, stride: 68, LR: 17x17

| Dataset | Scale              | PSNR(91-images/paper/ImageNet/paper)  | SSIM(91-images/paper/ImageNet/paper) |
| ------- | ------------------ | ------------------------------------- | ------------------------------------ |
| Set5    | x2<br />x3<br />x4 | 36.35/-<br />32.32/32.75<br />30.66/- |                                      |
| Set14   | x2<br />x3<br />x4 | 32.13/-<br />28.94/29.28<br />27.43/- |                                      |
| BSD200  | x2<br />x3<br />x4 | 31.08/-<br />28.16/28.55<br />26.93/- |                                      |
| BSD500  | x2<br />x3<br />x4 |                                       |                                      |

### DRCN

Dataset prepare: 91-image, 41x41, stride: 21

### DRRN

Dataset prepare: 91-image, BSD300 train set, 31x31, stride: 21, rotation: 0 90 180 270, flip: 0 1 2 3, uf 2 3 4, single model

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
| **3x**-PSNR |         |       |        |      |
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