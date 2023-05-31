# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import pdb
from sklearn.model_selection import train_test_split

## PyTorch dependencies
import torch
from torchvision import transforms
import torch.nn as nn
from torchgeo.transforms import AugmentationSequential
import torchgeo.transforms as T
from kornia import augmentation as K

## Local external libraries
from Datasets import preprocess

def get_transform(Network_parameters, input_size=224):
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']

    if Dataset == 'UCMerced':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(Network_parameters['resize_size']),
            transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(Network_parameters['center_size']),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
        
    elif Dataset == 'Eurosat_MSI':
        mean = torch.tensor(
        [
            1354.40546513,
            1118.24399958,
            1042.92983953,
            947.62620298,
            1199.47283961,
            1999.79090914,
            2369.22292565,
            2296.82608323,
            732.08340178,
            12.11327804,
            1819.01027855,
            1118.92391149,
            2594.14080798,
        ]
        )

        std = torch.tensor(
            [
                245.71762908,
                333.00778264,
                395.09249139,
                593.75055589,
                566.4170017,
                861.18399006,
                1086.63139075,
                1117.98170791,
                404.91978886,
                4.77584468,
                1002.58768311,
                761.30323499,
                1231.58581042,
            ]
        )

        data_transforms = {
            'train': AugmentationSequential(
                K.Resize(Network_parameters['resize_size']),
                K.RandomResizedCrop((input_size, input_size) , scale=(0.8, 1.0)),
                K.Normalize(mean=mean, std=std),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["image"],
            ),
            'test': AugmentationSequential(
                K.Resize(Network_parameters['resize_size']),
                K.CenterCrop(input_size),
                K.Normalize(mean=mean, std=std),
                data_keys=["image"],
            ),
        }

    elif Dataset == 'MSTAR': #Chips are fixed to be 128 by 128, may change in demo parameters
    #MSTAR is numpy arrays so custom center and random crop transforms are used in preprocessing
        mean = [0]
        std = [1]
        data_transforms = {
        'train': transforms.Compose([
            preprocess.CenterCrop(128),
            preprocess.RandomCrop(112), transforms.ToTensor(), 
            nn.ZeroPad2d(56),
        ]),
        'test': transforms.Compose([
            preprocess.CenterCrop(128), transforms.ToTensor(), 
            nn.ZeroPad2d(48),
        ]),
    }
        
    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset))
    
    return data_transforms, mean, std

