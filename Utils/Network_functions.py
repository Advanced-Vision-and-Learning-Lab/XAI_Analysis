# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy

## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models

## Local external libraries
from barbar import Bar
from Utils.pytorchtools import EarlyStopping
import pdb
from Utils.Focalnet import focalnet_tiny_srf
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def train_model(model, dataloaders, criterion, optimizer, device,
                num_epochs=25, scheduler=None):
    
    since = time.time()
    best_epoch = 0
    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []
    
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
   
    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
 
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode 
                else:
                    model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for idx, (inputs, labels, index) in enumerate(Bar(dataloaders[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    index = index.to(device)
        
                    # zero the parameter gradients
                    optimizer.zero_grad()
                  
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        
                        #Backward produces 2 losses
                        loss = criterion(outputs, labels.long()).mean()
                     
                        _, preds = torch.max(outputs, 1)
        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                          
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds.data == labels.data)
            
                epoch_loss = running_loss / (len(dataloaders[phase].sampler))
                epoch_acc = running_corrects.double().cpu().numpy() / (len(dataloaders[phase].sampler))
                
                if phase == 'train':
                    if scheduler is not None:
                        scheduler.step()
                    train_error_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
                if phase == 'val':
                    valid_loss = epoch_loss
                    val_error_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)
    
                print()
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
               
             
           #Check for early stopping (end training)
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print()
                print("Early stopping")
                break
            
            if torch.isnan(torch.tensor(valid_loss)):
                print()
                print('Loss is nan')
                break
         
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Val Acc: {:4f}'.format(best_acc))
        print()

    except:
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        
        # Return losses as dictionary
        train_loss = train_error_history
        
        val_loss = val_error_history
     
        #Return training and validation information
        train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                     'train_acc_track': train_acc_history, 
                      'train_error_track': train_loss,'best_epoch': best_epoch}
       
        print('Saved interrupt')
        return train_dict

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return losses as dictionary
    train_loss = train_error_history
    
    val_loss = val_error_history
 
    #Return training and validation information
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_loss,'train_acc_track': train_acc_history, 
                  'train_error_track': train_loss,'best_epoch': best_epoch}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
     
def test_model(dataloader,model,criterion,device,model_weights=None):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    running_loss = 0.0
    
    if model_weights is not None:
        model.load_state_dict(model_weights)
        
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels, index) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            #Run data through best model
            outputs = model(inputs)
         
            #Make model predictions
            _, preds = torch.max(outputs, 1)
            
            #Compute loss
            loss = criterion(outputs, labels.long()).mean()

            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
            #Keep track of correct predictions
            running_corrects += torch.sum(preds == labels.data)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            
    epoch_loss = running_loss / (len(dataloader.sampler))
    test_acc = running_corrects.double() / (len(dataloader.sampler))
    
    test_loss = {'total': epoch_loss}
    
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index[1:],
                 'test_acc': np.round(test_acc.cpu().numpy()*100,2),
                 'test_loss': test_loss}
    
    return test_dict

    
def initialize_model(model_name, num_classes,feature_extract=False,
                     use_pretrained=False, channels = 3):
    
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0
  
    #Select backbone architecture
    if model_name == "convnext":
        model_ft = models.convnext_tiny(pretrained=use_pretrained,
                                        weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        set_parameter_requires_grad(model_ft, feature_extract)
        if not(channels == 3):
            model_ft.features[0][0] = nn.Conv2d(channels, 96, kernel_size=(4,4),stride=(4,4))
        
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(model_ft.classifier[2].in_features, num_classes)
        input_size = 224

    elif model_name == "vit":
        model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        if not(channels == 3):
             model_ft.conv_proj = nn.Conv2d(channels, 768, kernel_size=(16,16),stride=(16,16)) 
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "focalnet":
        input_size = 224
        model_ft = focalnet_tiny_srf(pretrained = True, img_size = input_size, in_chans = channels)
        set_parameter_requires_grad(model_ft, feature_extract)
        if not(channels == 3):
            model_ft.patch_embed.proj = nn.Conv2d(channels, 96, kernel_size=(4,4),stride=(4,4))
        num_ftrs = model_ft.head.in_features        
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise RuntimeError('{} not implemented'.format(model_name))
    return model_ft, input_size

