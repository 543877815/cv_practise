import random
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.datasets import VOCDetection
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
from PIL import Image
import xml.etree.ElementTree as ET


def voc_collate_fn(batch_lst, reshape_size=224):
    # class_to_idx = {'aeroplane': 0, 'bicycle': first, 'bird': 2, 'boat': 3, 'bottle': 4,
    #                 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
    #                 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    #                 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    idx_to_class = {i: c for c, i in class_to_idx.items()}

    preprocess = transforms.Compose([
        transforms.Resize((reshape_size, reshape_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = len(batch_lst)

    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)

    max_num_box = max(len(batch_lst[i][1]['annotation']['object']) \
                      for i in range(batch_size))

    box_batch = torch.Tensor(batch_size, max_num_box, 5).fill_(-1.)
    w_list = []
    h_list = []
    img_id_list = []

    for i in range(batch_size):
        img, ann = batch_lst[i]
        w_list.append(img.size[0])  # image width
        h_list.append(img.size[1])  # image height
        img_id_list.append(ann['annotation']['filename'])
        img_batch[i] = preprocess(img)
        all_bbox = ann['annotation']['object']
        if type(all_bbox) == dict:  # inconsistency in the annotation file
            all_bbox = [all_bbox]
        for bbox_idx, one_bbox in enumerate(all_bbox):
            bbox = one_bbox['bndbox']
            obj_cls = one_bbox['name']
            box_batch[i][bbox_idx] = torch.Tensor([float(bbox['xmin']), float(bbox['ymin']),
                                                   float(bbox['xmax']), float(bbox['ymax']), class_to_idx[obj_cls]])

    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)

    return img_batch, box_batch, w_batch, h_batch, img_id_list


class VocDatasetDetection(VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

        grid (int): the size of grid in YOLOv1.
        bbnd (int): the size of bounding boxes per grid in YOLOv1.
        classes (int): the size of class in the dataset, which is 20 in voc2012 by default.
        resize (tuple): the size of resize of image.
    """

    def __init__(self,
                 root: str,
                 year: str = "2012",
                 image_set: str = "train",
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 grid: int = 7,
                 bbnd: int = 2,
                 classes: int = 20,
                 resize: tuple = (224, 224),
                 class_to_idx: dict = None
                 ):
        super(VocDatasetDetection, self).__init__(root, year, image_set, download,
                                                  transform, target_transform, transforms)
        self.grid = grid
        self.bbnd = bbnd
        self.classes = classes
        self.resize = resize
        if class_to_idx is None:
            self.class_to_idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                                 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
                                 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
                                 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        print(img.shape, target)
        img_size_before = [[int(x['depth']), int(x['width']), int(x['height'])] for x in target['annotation']['size']]
        # img_size_scalar = img_size_before / img.shape
        objects_info = target['annotation']['object']
        objects = [[self.class_to_idx[x['name']],
                    x['bndbox']['xmin'],
                    x['bndbox']['xmax'],
                    x['bndbox']['ymin'],
                    x['bndbox']['ymax']] for x in objects_info]


        return img, target

    def __len__(self) -> int:
        return len(self.images)
