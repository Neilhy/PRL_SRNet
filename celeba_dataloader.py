import os
import glob
import numpy as np
import torch
import random
import re

import cv2, datetime
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import multiprocessing

import joint_transforms


class CelebADataset(Dataset):
    """
    lr_dir: low-resolution
    hr_dir: high-resolution
    """

    def __init__(self, transforms_=None):
        self.transform = transforms_
        self.file_list_lr = []
        self.file_list_hr = []
        self.file_len = len(self.file_list_lr)

    def __getitem__(self, index):
        filename_lr = self.file_list_lr[index % self.file_len]
        filename_hr = self.file_list_hr[index % self.file_len]
        
        img_lr = Image.open(filename_lr) # 16 x 16
        img_hr = Image.open(filename_hr) # 64 x 64
        if img_lr.mode != 'RGB':
            img_lr = img_lr.convert('RGB')
        if img_hr.mode != 'RGB':
            img_hr = img_hr.convert('RGB')
        
        joint_transform = joint_transforms.RandomHorizontallyFlip()
        img_lr, img_hr = joint_transform(img_lr, img_hr)
        return self.transform(img_lr), self.transform(img_hr)

    def __len__(self):
        return self.file_len
