"""
MSTAR dataset object
Code modified from: https://github.com/jangsoopark/AConvNet-pytorch
"""
import numpy as np

from skimage import io
import torch
import tqdm

import json
import glob
import os
import pdb


class MSTAR_Dataset(torch.utils.data.Dataset):

    def __init__(self, path, name='soc', is_train=False, transform=None):
        self.is_train = is_train
        self.name = name

        self.images = []
        self.targets = []
        self.serial_number = []
        self.target_types = []
        # self.classes = ['2S1', 'BRDM2', 'ZSU234','T72']
        self.classes = []

        self.transform = transform
        self._load_data(path)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        _image = self.images[idx]
        _label = self.targets[idx]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label, idx

    def _load_data(self, path):
        mode = 'train' if self.is_train else 'test'

        image_list = glob.glob(os.path.join(path, f'{self.name}/{mode}/*/*.npy'))
        label_list = glob.glob(os.path.join(path, f'{self.name}/{mode}/*/*.json'))
        image_list = sorted(image_list, key=os.path.basename)
        label_list = sorted(label_list, key=os.path.basename)

        # for image_path, label_path in tqdm.tqdm(zip(image_list, label_list), 
        #                                         desc=f'load {mode} data set'):
        for image_path, label_path in zip(image_list, label_list):
            self.images.append(np.load(image_path))

            with open(label_path, mode='r', encoding='utf-8') as f:
                _label = json.load(f)

            self.targets.append(_label['class_id'])
            self.serial_number.append(_label['serial_number'])
            self.target_types.append(_label['target_type'])

        # #Get unique values for classes
        self.classes = np.unique(self.target_types)