# -*- coding: utf-8 -*-
"""
Parameters for XAI experiments
"""
import os
import sys

def Parameters(args):
    
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results

    #use xai interpretability
    xai = args.xai
    
    #Location to store trained models
    #Always add slash (/) after folder name
    folder = args.folder
    
    #Select dataset
    data_selection = args.data_selection
    Dataset_names = {1:'UCMerced', 2:'Eurosat_MSI', 3:'MSTAR'}
    
    #Flag for feature extraction. False, train whole model. True, only update
    #Flag to use pretrained model from ImageNet or train from scratch (default: True)
    feature_extraction = args.feature_extraction
    use_pretrained = args.use_pretrained
    add_bn = True
    scale = 5
    
    #Set learning rate for new and pretrained (pt) layers
    lr = args.lr
    
    #For no padding, set 0. If padding is desired,
    #enter amount of zero padding to add to each side of image
    #(did not use padding in paper, recommended value is 0 for padding)
    padding = 0
    
    #Apply rotation to test set (did not use in paper)
    #Set rotation to True to add rotation, False if no rotation (used in paper)
    #Recommend values are between 0 and 25 degrees
    #Can use to test robustness of model to rotation transformation
    rotation = False
    degrees = 25
    
    #Set step_size and decay rate for scheduler
    #In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 10
    gamma = .1
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs,
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size}
    num_epochs = args.num_epochs
    
    #Resize the image before center crop. Recommended values for resize is 256 (used in paper), 384,
    #and 512 (from http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf)
    #Center crop size is recommended to be 256.
    resize_size = args.resize_size
    center_size = 224
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = True
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = 0

    #Visualization of results parameters
    #Visualization parameters for figures
    fig_size = 12
    font_size = 16
    
    #Flag for TSNE visuals, set to True to create TSNE visual of features
    #Set to false to not generate TSNE visuals
    #Number of images to view for TSNE (defaults to all training imgs unless
    #value is less than total training images).
    TSNE_visual = False
    Num_TSNE_images = 5000
    
    #Set to True if more than one GPU was used 
    #False for UCMerced dataset only
    #True for EuroSAT and MSTAR dataset
    Parallelize_model = True
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    if feature_extraction:
        mode = 'Feature_Extraction'
    else:
        mode = 'Fine_Tuning'
    
    #Location of texture datasets
    Data_dirs = {'UCMerced' : './Datasets/UCMerced/',
                 'Eurosat_MSI': './Datasets/Eurosat_MSI',
                 'MSTAR': 'Datasets/MSTAR'}
    
    #Backbone architecture
    #Options are resnet18, resnet50, resnet50_wide, resnet50_next, VGG16, inception_v3
    # densenet161, inception_v3
    Model_name = args.model
    
    #channels in each dataset
    channels = {'UCMerced': 3,
                'Eurosat_MSI': 13,
                'MSTAR': 2}
    
    #Number of classes in each dataset
    num_classes = {'UCMerced': 21,
                   'Eurosat_MSI': 10,
                   'MSTAR': 4}
    
    #Number of runs and/or splits for each dataset
    Splits = {'UCMerced': 3,
              'Eurosat_MSI': 3,
              'MSTAR': 3}
    
    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    
    #Return dictionary of parameters
    Params = {'save_results': save_results,'folder': folder,
            'Dataset': Dataset, 'data_dir': data_dir,
            'num_workers': num_workers, 'mode': mode,
            'lr': lr,'step_size': step_size,'gamma': gamma, 
            'batch_size' : batch_size, 'num_epochs': num_epochs, 
            'resize_size': resize_size, 'center_size': center_size, 
            'padding': padding,'Model_name': Model_name, 'num_classes': num_classes, 
            'Splits': Splits, 'feature_extraction': feature_extraction,
            'use_pretrained': use_pretrained,
            'xai': xai,
            'add_bn': add_bn, 'pin_memory': pin_memory, 'scale': scale,
            'degrees': degrees, 'rotation': rotation, 
            'TSNE_visual': TSNE_visual,
            'Parallelize': Parallelize_model,'Num_TSNE_images': Num_TSNE_images,
            'fig_size': fig_size,'font_size': font_size,
            'channels': channels}
    
    return Params
