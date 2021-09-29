from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random

class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split == "val":
            file_list = os.listdir(self.root + 'test_patch_npy/')
            self.images_path = self.root + 'test_patch_npy/'
            label_list = os.listdir(self.root + 'test_mask_patch_npy/')
            self.labels_path = self.root + 'test_mask_patch_npy/'
        elif self.split == "train_supervised":
            file_list = os.listdir(self.root + 'train_patch_npy/')
            self.images_path = self.root + 'train_patch_npy/'
            label_list = os.listdir(self.root + 'train_mask_patch_npy/')
            self.labels_path = self.root + 'train_mask_patch_npy/'
        elif self.split == "train_unsupervised":
            file_list = os.listdir(self.root + 'train_patch_npy/')#_140
            self.images_path = self.root + 'train_patch_npy/'#_140
            label_list = None
        else:
            raise ValueError(f"Invalid split name {self.split}")

        # file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        # self.files, self.labels = list(zip(*file_list))
        random.shuffle(file_list)
        if self.split == 'val':
            self.files = file_list
        elif self.split == 'train_unsupervised':
            self.files = file_list[:int(len(file_list)*0.1)]   #self.file_list*0.05
        else:
            self.files = file_list

        # self.files = file_list
        self.labels = label_list

    def _load_data(self, index):
        image_path = os.path.join(self.images_path, self.files[index])
        image = np.load(image_path)
        image = np.transpose(image, (1, 2, 0))
        image_id = self.files[index][:-4]
        if self.labels:
            if self.use_weak_lables:
                label_path = os.path.join(self.weak_labels_output, image_id+"_mask.npy")
            else:
                label_path = os.path.join(self.labels_path, image_id + '_mask.npy')
            label = np.load(label_path)
            label = np.transpose(label, (1, 2, 0))
        else:
            label = np.zeros_like(image)

        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        return image, label, image_id

class VOC(BaseDataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = 0
        self.STD = 255
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        # kwargs['ignore_index'] = 0
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = VOCDataset(**kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
