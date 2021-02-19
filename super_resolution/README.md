# Super Resolution

## Methods

### SRCNN



### FSRCNN



### VDSR



## Results

The method transforms RGB to YCrCb is mainly refer to matlab' s `rgb2ycrcb` function. The PSNR is calculated only on the y channel on the [YCrCb](https://en.wikipedia.org/wiki/YCbCr) color space, and the detail Implementation is refer to the code in [SelfExSR](https://github.com/jbhuang0604/SelfExSR/blob/master/quant_eval/compute_difference.m), which crops some margin pixels during testing in order to get the similar result as the paper.



Results on Set 5

| Scale       | BicuBic | SRCNN | FSRCNN | VDSR |
| ----------- | ------- | ----- | ------ | ---- |
| **2x**-PSNR | 33.64   |       |        |      |
| **3x**-PSNR | 30.39   |       |        |      |
| **4x**-PSNR | 28.42   |       |        |      |
| **2x**-SSIM | 0.9292  |       |        |      |
| **3x**-SSIM | 0.8678  |       |        |      |
| **4x**-SSIM | 0.8101  |       |        |      |
| **2x**-IFC  | 5.72    |       |        |      |
| **3x**-IFC  | 3.45    |       |        |      |
| **4x**-IFC  | 2.28    |       |        |      |

Results on Set 14

| Scale       | BicuBic | SRCNN | FSRCNN | VDSR |
| ----------- | ------- | ----- | ------ | ---- |
| **2x**-PSNR | 30.22   |       |        |      |
| **3x**-PSNR | 27.53   |       |        |      |
| **4x**-PSNR | 25.99   |       |        |      |
| **2x**-SSIM | 0.8683  |       |        |      |
| **3x**-SSIM | 0.7737  |       |        |      |
| **4x**-SSIM | 0.7023  |       |        |      |
| **2x**-IFC  | 5.74    |       |        |      |
| **3x**-IFC  | 3.33    |       |        |      |
| **4x**-IFC  | 2.18    |       |        |      |

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