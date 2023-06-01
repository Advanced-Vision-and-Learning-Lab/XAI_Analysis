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
from pytorch_metric_learning import samplers

import ssl
## PyTorch dependencies
import torch
## Local external libraries
from Datasets.Pytorch_Datasets import *
from Datasets.Get_transform import *
from Datasets import preprocess
from Datasets import loader

def collate_fn_train(data):
    data, labels, index = zip(*data)
    data = torch.stack(data)
    labels = torch.stack(labels)
    data = data_transforms["train"]({"image":data.float()})
    index = torch.Tensor(index)
    return data["image"].float(), labels.float(), index

def collate_fn_test(data):
    data, labels, index = zip(*data)
    data = torch.stack(data)
    labels = torch.stack(labels)
    data = data_transforms["test"]({"image":data.float()})
    index = torch.Tensor(index)
    return data["image"].float(), labels.float(), index

def Prepare_DataLoaders(Network_parameters, split,input_size=224, view_results = True):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    dataset_sampler = {}

    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    global data_transforms
    data_transforms, mean, std = get_transform(Network_parameters, input_size=input_size)
    Network_parameters['mean'] = mean
    Network_parameters['std'] = std
       
    if Dataset == 'UCMerced': #See people also use .5, .5 for normalization
        train_dataset = UCMerced_Index(root = data_dir,split='train', transform = data_transforms['train'], download=True)
        test_dataset = UCMerced_Index(data_dir,split='test', transform = data_transforms['test'], download=True)
        val_dataset = UCMerced_Index(data_dir,split='val', transform = data_transforms['test'], download=True)
        
    elif Dataset == 'Eurosat_MSI': #See people also use .5, .5 for normalization
        train_dataset = Eurosat_MSI(root = data_dir,split='train', transform = None, download=True)
        test_dataset = Eurosat_MSI(data_dir,split='test', transform = None, download=True)
        val_dataset = Eurosat_MSI(data_dir,split='val', transform = None, download=True)
        
    elif Dataset == 'MSTAR':
        #Load training and testing data
        #Break up training into train and val paritions (default is 80/20)
        train_dataset = loader.MSTAR_Dataset(data_dir, name='eoc-1-t72-a64', is_train=True,
        transform=data_transforms['test'])
        
        X = np.arange(0,len(train_dataset))
        y = train_dataset.targets
        #Set random state to keep the data the same order for each model
        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=.2,stratify=y,
                                                          random_state=42)
        
        train_dataset = torch.utils.data.Subset(loader.MSTAR_Dataset(data_dir, name='eoc-1-t72-a64', is_train=True,
        transform=data_transforms['train']),X_train)
        val_dataset =  torch.utils.data.Subset(loader.MSTAR_Dataset(data_dir, name='eoc-1-t72-a64', is_train=True,
        transform=data_transforms['test']),X_val)
        test_dataset = loader.MSTAR_Dataset(data_dir, name='eoc-1-t72-a64', is_train=False,
        transform=data_transforms['test'])
        
        # print('Number of Training Samples: {}'.format(len(train_dataset)))
        # print('Number of Val Samples: {}'.format(len(val_dataset)))
        # print('Number of Test Samples: {}'.format(len(test_dataset)))

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 
    if view_results:
        labels = test_dataset.targets
        classes = test_dataset.classes
        #m is the number of samples taken from each class
        m = 10
        #In our paper, batch_size
            #UCMerced - 210
            #EuroSAT - 100
            #MSTAR - 40
        batch_size = m*len(classes)
        sampler = samplers.MPerClassSampler(labels, m, batch_size, length_before_new_iter=100000)
        #retain sampler = None for 'train' and 'val' data splits
        dataset_sampler = {'train': None, 'test': sampler, 'val': None}
        Network_parameters["batch_size"]["test"] = batch_size

    else:
        dataset_sampler = {'train': None, 'test': None, 'val': None}
    #Create dataloaders
    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    collate_fn = {'train': collate_fn_train, 'val': collate_fn_test, 'test': collate_fn_test}
    

    #Collate function is used only for EuroSAT and MSTAR
    #Compatible input size for Kornia augmentation
    if Dataset == "UCMerced" or Dataset == "MSTAR":
    # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        shuffle=not(view_results),
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        sampler = dataset_sampler[x])
                                                        for x in ['train', 'val','test']}
        
    else:
            # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        shuffle=not(view_results),
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        collate_fn = collate_fn[x],
                                                        sampler = dataset_sampler[x])
                                                        for x in ['train', 'val','test']}

    return dataloaders_dict