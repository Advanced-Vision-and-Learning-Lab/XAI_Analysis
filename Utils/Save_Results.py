# -*- coding: utf-8 -*-
"""
Save results from training/testing model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import os
import pickle

## PyTorch dependencies
import torch


def save_results(train_dict, test_dict, split, Network_parameters, num_params,
                 model_ft):
    # Baseline model
    filename = '{}/{}/{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                             Network_parameters['mode'],
                                             Network_parameters['Dataset'],
                                             Network_parameters['Model_name'],
                                             split+1)

    if not os.path.exists(filename):
        os.makedirs(filename)
        
    #Will need to update code to save everything except model weights to
    # dictionary (use torch save)
    #Save test accuracy
    with open((filename + 'Test_Accuracy.txt'), "w") as output:
        output.write(str(test_dict['test_acc']))
    
    #Save training and testing dictionary, save model using torch
    torch.save(train_dict['best_model_wts'], filename + 'Best_Weights.pt')
    

    #Remove model from training dictionary
    train_dict.pop('best_model_wts')
    output_train = open(filename + 'train_dict.pkl','wb')
    pickle.dump(train_dict,output_train)
    output_train.close()
    
    output_test = open(filename + 'test_dict.pkl','wb')
    pickle.dump(test_dict,output_test)
    output_test.close()

        
        
    

 