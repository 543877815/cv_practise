VDSR
data prepare
python data_aug.py --number 291 --use_bicubic --width 41 --height 41 --stride 41 -uf 2 3 4 --scales 1.0 0.7 0.5 --rotations 0 90 180 270 --flips 0 1 2 3 \
        --input /data/data/291-images/ --single_y --use_h5py --output /data/data/super_resolution/data_for_VDSR/train.h5
#        --input F:\\cache\\data\\291-image\\HR --single_y --use_h5py --output F:\\cache\\data\\data_for_VDSR\\train.h5

train
distributed
python -m torch.distributed.launch --nproc_per_node=2 main.py --configs configs/vdsr.yaml
common
python main.py --configs configs/vdsr.yaml

FSRCNN
data prepare
python data_aug.py --number 1 --width 20 --height 20 --stride 2 -uf 2 \
				   --rotations 0 90 180 270 --scales 1.0 0.9 0.8 0.7 0.6 \
           --input /data/data/91-images/data --single_y --use_h5py \
           --output /data/data/super_resolution/data_for_FSRCNN/test_x2.h5