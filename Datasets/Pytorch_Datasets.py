# -*- coding: utf-8 -*-
"""
Return index of built in Pytorch datasets 
"""
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import pdb
import torchgeo
import torch
import torchvision.transforms as T
import kornia.augmentation as K
from torchgeo.datasets import EuroSAT

device = "cuda" if torch.cuda.is_available() else "cpu"

          
class UCMerced_Index(Dataset):
    def __init__(self, root, split='train',transform=None, download=True):  
        
        self.transform = transform
        self.split = split
        self.images = torchgeo.datasets.UCMerced(root,split,transforms=transform,
                                           download=download)
        self.targets = self.images.targets        
        self.classes = self.images.classes
       
    def __getitem__(self, index):
        image, target = self.images._load_image(index)
                
        if self.transform is not None:
            to_pil = T.ToPILImage()
            image = to_pil(image)
            image = self.transform(image)
        return image, target, index

    def __len__(self):
        return len(self.images)
    
class Eurosat_MSI(Dataset):
    def __init__(self, root, split='train',transform=None, download=True):
        
        self.transform = transform
        self.split = split
        self.images = torchgeo.datasets.EuroSAT(root,split,transforms=transform, download=download)
        self.band_indices =self.images.band_indices
        self.targets = self.images.targets       
        self.classes = self.images.classes
       
    def __getitem__(self, index):
        image, target = self.images._load_image(index)
        sample = {"image": image, "label": target}
            
        if self.transform is not None:
            image = self.transform(image)
        return image, target, index

    def __len__(self):
        return len(self.images)
    



