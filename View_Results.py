# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
import pickle
import argparse
import pdb
from XAI_methods.get_explanations import get_metrics, plot_avg_spyder_plot


## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Generate_TSNE_visual import Generate_TSNE_visual
from Demo_Parameters import Parameters
from Utils.Network_functions import initialize_model
from Prepare_Data import Prepare_DataLoaders
from Utils.Confusion_mats import plot_confusion_matrix, plot_avg_confusion_matrix
from Utils.Generate_Learning_Curves import Plot_Learning_Curves
from Datasets.Pytorch_Dataset_Names import Get_Class_Names


plt.ioff()

def main(Params):

    # Location of experimental results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fig_size = Params['fig_size']
    font_size = Params['font_size']
    
    # Set up number of runs and class/plots names
    NumRuns = Params['Splits'][Params['Dataset']]
    plot_name = Params['Dataset'] + ' Test Confusion Matrix'
    avg_plot_name = Params['Dataset'] + ' Test Average Confusion Matrix'
    class_names = Get_Class_Names(Params['Dataset'], Params['data_dir'])
    
    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name']
    
    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    # Initialize arrays for results
    cm_stack = np.zeros((len(class_names), len(class_names)))
    cm_stats = np.zeros((len(class_names), len(class_names), NumRuns))
    df_metrics_avg_rank = []
    df_metrics_avg_score = []
    FDR_scores = np.zeros((len(class_names), NumRuns))
    log_FDR_scores = np.zeros((len(class_names), NumRuns))
    accuracy = np.zeros(NumRuns)
    MCC = np.zeros(NumRuns)
    
    for split in range(0, NumRuns):
        # torch.manual_seed(split)
        # np.random.seed(split)
        # np.random.seed(split)
        # torch.cuda.manual_seed(split)
        # torch.cuda.manual_seed_all(split)
        # torch.manual_seed(split)
        
        sub_dir = '{}/{}/{}/{}/Run_{}/'.format(Params['folder'],
                                                 Params['mode'],
                                                 Params['Dataset'],
                                                 Params['Model_name'],
                                                 split+1)
        
        # Load training and testing files (Python)
        train_pkl_file = open(sub_dir + 'train_dict.pkl', 'rb')
        train_dict = pickle.load(train_pkl_file)
        train_pkl_file.close()
    
        test_pkl_file = open(sub_dir + 'test_dict.pkl', 'rb')
        test_dict = pickle.load(test_pkl_file)
        test_pkl_file.close()
    
        # Remove pickle files
        del train_pkl_file, test_pkl_file
    
    
        # Initialize the histogram model for this run
        model, input_size = initialize_model(model_name, num_classes,
                                                feature_extract=Params['feature_extraction'],
                                                use_pretrained=Params['use_pretrained'],
                                                channels = Params["channels"][Dataset])
    
        # Set device to cpu or gpu (if available)
        device_loc = torch.device(device)
        # Generate learning curves
        Plot_Learning_Curves(train_dict['train_acc_track'],
                             train_dict['train_error_track'],
                             train_dict['val_acc_track'],
                             train_dict['val_error_track'],
                             train_dict['best_epoch'],
                             sub_dir)
    
        # If parallelized, need to set change model
        if Params['Parallelize']:
            model = nn.DataParallel(model)
           
    
        model.load_state_dict(torch.load(sub_dir + 'Best_Weights.pt', map_location=device_loc))
        model = model.to(device)
    
        dataloaders_dict = Prepare_DataLoaders(Params, split,
                                                input_size=input_size)
        
        #XAI metrics
        if (Params['xai']):
            df_metric, df_metric_ranks = get_metrics(dataloaders_dict['test'], Dataset, model, device, 
                                                     sub_dir, Params, Params['Parallelize'])
            df_metrics_avg_rank.append(np.expand_dims(df_metric_ranks.to_numpy(), axis = -1)) 
            df_metrics_avg_score.append(np.expand_dims(df_metric.to_numpy(), axis = -1))
            df_metric.to_csv((sub_dir + 'xai_metric.csv'))
            df_metric_ranks.to_csv((sub_dir + 'xai_metric_ranks.csv'))

    
        if (Params['TSNE_visual']):
            print("Initializing Datasets and Dataloaders...")
    
            dataloaders_dict = Prepare_DataLoaders(params, split,
                                                    input_size=input_size)
            print('Creating TSNE Visual...')
            
            #Remove fully connected layer
            if Params['Parallelize']:
                try:
                    model.module.fc = nn.Sequential()
                except:
                    model.module.classifier = nn.Sequential()
            else:
                try:
                    model.fc = nn.Sequential()
                except:
                    model.classifier = nn.Sequential()
    
            # Generate TSNE visual
            FDR_scores[:, split], log_FDR_scores[:, split] = Generate_TSNE_visual(
                dataloaders_dict,
                model, sub_dir, device, class_names)
            
        # Create CM for testing data
        cm = confusion_matrix(test_dict['GT'], test_dict['Predictions'])
        
        # Create classification report
        report = classification_report(test_dict['GT'], test_dict['Predictions'],
                                       target_names=class_names, output_dict=True)
        
        # Convert to dataframe and save as .CSV file
        df = pd.DataFrame(report).transpose()
        
        # Save to CSV
        df.to_csv((sub_dir + 'Classification_Report.csv'))
    
        # Confusion Matrix
        np.set_printoptions(precision=2)
        fig4, ax4 = plt.subplots(figsize=(fig_size, fig_size))
        plot_confusion_matrix(cm, classes=class_names, title=plot_name, ax=ax4,
                              fontsize=font_size)
        fig4.savefig((sub_dir + 'Confusion Matrix.png'), dpi=fig4.dpi)
        plt.close(fig4)
        cm_stack = cm + cm_stack
        cm_stats[:, :, split] = cm
    
        # Get accuracy of each cm
        accuracy[split] = 100 * sum(np.diagonal(cm)) / sum(sum(cm))
        # Write to text file
        with open((sub_dir + 'Accuracy.txt'), "w") as output:
            output.write(str(accuracy[split]))
    
        # Compute Matthews correlation coefficient
        MCC[split] = matthews_corrcoef(test_dict['GT'], test_dict['Predictions'])
    
        # Write to text file
        with open((sub_dir + 'MCC.txt'), "w") as output:
            output.write(str(MCC[split]))
        directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'
    
        print('**********Run ' +  str(split + 1) + '  Finished**********')
    
    directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'
    np.set_printoptions(precision=2)
    fig5, ax5 = plt.subplots(figsize=(fig_size, fig_size))
    plot_avg_confusion_matrix(cm_stats, classes=class_names,
                              title=avg_plot_name, ax=ax5, fontsize=font_size)
    fig5.savefig((directory + 'Average Confusion Matrix.png'), dpi=fig5.dpi)
    plt.close()
    
    #Average rank of the metric scores
    if (Params['xai']):
        #Calculation of ranks across the average dataframe
        df_metrics_overall_rank = np.concatenate(df_metrics_avg_rank, axis = -1)
        df_metrics_avg_rank = np.mean(df_metrics_overall_rank, axis = -1)
        df_metrics_std_rank = np.std(df_metrics_overall_rank, axis = -1)

        #Calculation of average scores across the dataframe
        df_metrics_overall_score = np.concatenate(df_metrics_avg_score, axis = -1)
        df_metrics_avg_score = np.mean(df_metrics_overall_score, axis = -1)
        columns = df_metric.columns
        index = df_metric.index
        spyder_df_metrics_avg_score = pd.DataFrame(df_metrics_avg_score,columns = columns, index = index)
        spyder_df_metrics_avg_score.to_csv((directory + 'Avg metric scores.csv'))


        eps = 10e-6
        df = spyder_df_metrics_avg_score
        df_normalised = df.loc[:, ~df.columns.isin(['Robustness', 'Randomization'])].apply(lambda x: x / (x.max() + eps))
        df_normalised["Robustness"] = df["Robustness"].min()/(df["Robustness"].values + eps)
        df_normalised["Randomisation"] = df["Randomization"].min()/(df["Randomization"].values + eps)
        df_normalised_rank = df_normalised.rank()
        pd.DataFrame(df_normalised_rank,columns = columns, index = index).to_csv((directory + 'Average rank.csv'))
        plot_avg_spyder_plot(pd.DataFrame(df_normalised_rank), pd.DataFrame(df_metrics_std_rank), directory)

    # Write to text file
    with open((directory + 'Overall_Accuracy.txt'), "w") as output:
        output.write('Average accuracy: ' + str(np.mean(accuracy)) + ' Std: ' + str(np.std(accuracy)))
    
    # Write to text file
    with open((directory + 'Overall_MCC.txt'), "w") as output:
        output.write('Average MCC: ' + str(np.mean(MCC)) + ' Std: ' + str(np.std(MCC)))
    
    # Write to text file
    with open((directory + 'training_Overall_FDR.txt'), "w") as output:
        output.write('Average FDR: ' + str(np.mean(FDR_scores, axis=1))
                     + ' Std: ' + str(np.std(FDR_scores, axis=1)))
    with open((directory + 'training_Overall_Log_FDR.txt'), "w") as output:
        output.write('Average FDR: ' + str(np.mean(log_FDR_scores, axis=1))
                     + ' Std: ' + str(np.std(log_FDR_scores, axis=1)))
    
    # Write list of accuracies and MCC for analysis
    np.savetxt((directory + 'List_Accuracy.txt'), accuracy.reshape(-1, 1), fmt='%.2f')
    np.savetxt((directory + 'List_MCC.txt'), MCC.reshape(-1, 1), fmt='%.2f')
    
    np.savetxt((directory + 'training_List_FDR_scores.txt'), FDR_scores, fmt='%.2E')
    np.savetxt((directory + 'training_List_log_FDR_scores.txt'), log_FDR_scores, fmt='%.2f')
    plt.close("all")

def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments(default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/Test_Code/',
                        help='Location to save models')
    parser.add_argument('--data_selection', type=int, default=3,
                        help='Dataset selection:  1:UCMerced, 2:Eurosat_MSI, 3:MSTAR')
    parser.add_argument('--feature_extraction', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected/encoder parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--xai', default=True, action=argparse.BooleanOptionalAction,
                        help='enables xai interpretability')
    parser.add_argument('--Parallelize', default=True, action=argparse.BooleanOptionalAction,
                        help='enables parallel functionality')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--model', type=str, default='focalnet',
                        help='backbone architecture to use (default: 0.01)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)