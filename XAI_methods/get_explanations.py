
"""
Modified Code from: https://github.com/understandable-machine-intelligence-lab/Quantus/tree/main

"""
import quantus
import numpy as np
import pandas as pd
from XAI_methods.get_spyderplot import *
from XAI_methods.get_attributes import get_attributions
import matplotlib.pyplot as plt
import pdb
from captum.attr import *
from captum.attr import visualization as viz
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
import cv2
import warnings


# Run full quantification analysis!
def evaluate_xai_methods(
                         model, metrics: dict, 
                         xai_methods: dict, 
                         x_batch: np.ndarray, 
                         y_batch: np.ndarray,
                         s_batch: np.ndarray,
                         nr_samples: int = 1,
                         explain_func = None,
                         ):
    """Evaluate explanations."""
    
    results = {}
    for method, explainer in xai_methods.items():
        results[method] = {}
        for metric, metric_func in metrics.items():
            print(f"Evaluating the {metric} of {method} method...")
            
            # Get scores and append results.
            
            scores = metric_func(
                model=model,
                x_batch=x_batch[:nr_samples],
                y_batch=y_batch[:nr_samples],
                s_batch=s_batch[:nr_samples],
                a_batch = explainer[:nr_samples],
                explain_func=explain_func,
                explain_func_kwargs={"method": method}
            )
          
            results[method][metric] = scores #results[method][metric] = np.random.random(size=nr_test_samples)
    return results

def eurosat_to_rgb(x):
    x = x[[3, 2, 1], :, :].astype(float)
    x -= np.min(x)
    x /= np.max(x)
    return x

def get_magnitude(x):
    x = np.float32(x[0, :, :])
    x -= np.min(x)
    x /= np.max(x)
    return x

def inverse_normalize(array, mean=(0, 0, 0), std=(1, 1, 1)):
    array = np.multiply(array, std)  # Multiply by standard deviation
    array = np.add(array, mean)  # Add mean values
    return array

def show_attr(attr_map):
    viz.visualize_image_attr(
        np.transpose(attr_map, (1, 2, 0)),  # adjust shape to height, width, channels 
        method='heat_map',
        sign='all',
        show_colorbar=False
    )

def plot_XAI_methods(dataloaders, dataset, x_batch, y_batch, directory, Params):
    num_samples = len(dataloaders.dataset)
    num_samples_to_pick = 3
    random_indices = random.sample(range(num_samples), num_samples_to_pick)
    y_batch = y_batch.astype(int)

    # Plotting configs.
    colours_order = ["#d62728", "#FFA500", "#008080", "#008000", "#800080"]
    methods_order = ["Lime", "HiResCAM", "GradientShap", "Saliency", "Occlusion"]

    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False

    # Plot explanations!
    index = 1
    ncols = 1 + len(xai_methods)
    
    for index in [1,2]:
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(13, int(ncols)*3))
        if dataset == "Eurosat_MSI":
            image = x_batch[index]
            image = eurosat_to_rgb(image) #get rgb image
            trans_image = np.transpose(image, (1, 2, 0))
            image = inverse_normalize(trans_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        elif dataset == "MSTAR":
            image = x_batch[index]
            image = get_magnitude(image)
            
        else:
            image = x_batch[index]
            image = np.transpose(image, (1, 2, 0))
            image = inverse_normalize(image, Params["mean"], Params["std"])
            image = image.astype(np.float32)

        for i in range(ncols):
            xai = methods_order[i-1].split("(")[0].replace(" ", "").replace("\n", "")

            if i == 0:
                with warnings.catch_warnings():
                    axes[0].imshow(image, vmin=0.0, vmax=1.0)
                    axes[0].set_title(f"{dataloaders.dataset.classes[y_batch[index]][:19]}", fontsize=14)
                    axes[0].axis("off")
            else:
                if i == 1 or i == 2:
                    if "HiResCAM" in methods_order:
                       
                        if dataset == "MSTAR":
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        grayscale_cam = xai_methods[xai][index]
                        visualization = show_cam_on_image(image, grayscale_cam)
                        axes[i].imshow(visualization)

                else:
                    axes[i].imshow(quantus.normalise_func.normalise_by_negative(xai_methods[xai]
                                            [index].reshape(224, 224)), cmap="seismic")

                axes[i].set_title(f"{methods_order[i-1]}", fontsize=12)
                # Frame configs.
                axes[i].xaxis.set_visible([])
                axes[i].yaxis.set_visible([])
                axes[i].spines["top"].set_color(colours_order[i-1])
                axes[i].spines["bottom"].set_color(colours_order[i-1])
                axes[i].spines["left"].set_color(colours_order[i-1])
                axes[i].spines["right"].set_color(colours_order[i-1])
                axes[i].spines["top"].set_linewidth(5)
                axes[i].spines["bottom"].set_linewidth(5)
                axes[i].spines["left"].set_linewidth(5)
                axes[i].spines["right"].set_linewidth(5)

        fig.tight_layout()
        fig.savefig(directory +  "XAI_methods_" + str(index) + ".png", dpi=fig.dpi)
        plt.close()

def plot_avg_spyder_plot(df_metrics_avg_rank, df_metrics_std_rank, directory):
    data = [df_metrics_avg_rank.columns.values, (df_metrics_avg_rank.to_numpy())]
    theta = spyder_plot(len(data[0]), frame='polygon')
    spoke_labels = data.pop(0)
    colours_order = ["red", "darkorange", "royalblue", "darkgreen", "slateblue", "purple"]
    include_titles, include_legend = True, True


    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    for i, (d, method) in enumerate(zip(data[0], xai_methods)):
        line = ax.plot(theta, d, label=method, color=colours_order[i], linewidth=5.0)
        ax.fill(theta, d, alpha=0.15)

    # Set lables.
    if include_titles:
        ax.set_varlabels(labels=["Faithfulness", "Localization", "Complexity", "Axiomatic","Robustness","Randomization"])
    else:
        ax.set_varlabels(labels=[])
    ax.set_rgrids(np.arange(0, df_metrics_avg_rank.values.max() + 0.5), labels=[]) 

    # Put a legend to the right of the current axis.
    if include_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))


    fig.tight_layout()
    fig.savefig(directory + "spyderplot_metrics.png", dpi=fig.dpi)
    plt.close()

def get_metrics(dataloaders, dataset, model, device, dir, Params, parallel=False):
    # Initialise the Quantus evaluation metrics.
    metrics_avg = {
        "Robustness": quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
            abs=True,
            normalise=False,
            aggregate_func=np.nanmean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Faithfulness": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=224,
            perturb_baseline="black",
            perturb_func=quantus.baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=True,
            normalise=False,
            aggregate_func=np.nanmean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Localization": quantus.RelevanceRankAccuracy(
            abs=True,
            normalise=False,
            aggregate_func=np.nanmean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Complexity": quantus.Sparseness(
            abs=True,
            normalise=False,
            aggregate_func=np.nanmean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Randomization": quantus.ModelParameterRandomisation(
            layer_order="independent",
            similarity_func=quantus.ssim,
            return_sample_correlation=True,
            abs=True,
            normalise=False,
            aggregate_func=np.nanmean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Axiomatic":quantus.Completeness(
            abs=False,
            normalise=False,
            aggregate_func=np.nanmean,
            return_aggregate=True,
            disable_warnings=True,
        ),
    }

    # Define XAI methods to score.
    global xai_methods
    xai_methods, x_batch, y_batch, s_batch = get_attributions(dataloaders, dataset, model, device, Params, parallel)

    if dataset == "Eurosat_MSI":
        y_batch = y_batch.astype(int)
        model = model.to("cuda")
    
    elif dataset == "MSTAR":
        y_batch = y_batch.astype(int)
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    model.eval()

    #plotting XAI METHODS
    #plot_XAI_methods(dataloaders, dataset, x_batch, y_batch, dir, Params)
   
    def explainer_func(model, inputs, targets, **kwargs) -> np.ndarray:
        "Pseudo explainer function"
        
        return np.zeros((inputs.shape[0],inputs.shape[2],inputs.shape[3]))
    
  
    results = evaluate_xai_methods(model, metrics_avg,
                                xai_methods,
                                x_batch,
                                y_batch,
                                s_batch,
                                explain_func=explainer_func,
                                nr_samples=x_batch.shape[0])
    

    results_agg = {}
    for method in xai_methods:
        results_agg[method] = {}
        for metric, metric_func in metrics_avg.items():
            results_agg[method][metric] = np.mean(results[method][metric])

    df = pd.DataFrame.from_dict(results_agg)
    df = df.T.abs()

    # Take inverse ranking for Robustness, since lower is better.
    eps = 10e-6
    df_normalised = df.loc[:, ~df.columns.isin(['Robustness', 
                        'Randomization'])].apply(lambda x: x / (x.max() + eps))
    
    df_normalised["Robustness"] = df["Robustness"].min()/(df["Robustness"].values + eps)
    df_normalised["Randomization"] = df["Randomization"].min()/(df["Randomization"].values + eps)
    df_inverse = df_normalised.copy()  # Create a copy of the normalized DataFrame

    # Negative normalization for plotting
    df_inverse = df.loc[:, ~df.columns.isin(['Robustness', 
                        'Randomization'])].apply(lambda x: (x.max() - x) / (x.max() - x.min() + eps))

    # Reverse normalization for 'Robustness'
    df_inverse['Robustness'] = df['Robustness'].min() / (df_normalised['Robustness'].values + eps)

    # Reverse normalization for 'Randomisation'
    df_inverse['Randomization'] = df['Randomization'].min() / (
                        df_normalised['Randomization'].values + eps)
    df_normalised_rank = df_inverse.rank(ascending = False)

    # Make spyder graph!
    data = [df_normalised_rank.columns.values, (df_normalised_rank.to_numpy())]
    theta = spyder_plot(len(data[0]), frame='polygon')
    spoke_labels = data.pop(0)
    colours_order = ["red", "darkorange", "royalblue", "darkgreen", "slateblue", "purple"]
    include_titles, include_legend = True, True

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    for i, (d, method) in enumerate(zip(data[0], xai_methods)):
        line = ax.plot(theta, d, label=method, color=colours_order[i], linewidth=5.0)
        ax.fill(theta, d, alpha=0.15)

    # Set lables.
    if include_titles:
        ax.set_varlabels(labels=["Faithfulness", "Localization", "Complexity", "Axiomatic","Robustness","Randomization"])
    else:
        ax.set_varlabels(labels=[]) 
    ax.set_rgrids(np.arange(0, df_normalised_rank.values.max() + 0.5), labels=[]) 

    # Put a legend to the right of the current axis.
    if include_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    fig.savefig(dir + "spyderplot_metrics.png", dpi=fig.dpi)
    plt.close()
    
    df_inverse_normalised_rank = df_normalised.rank(ascending = False)
    return df, df_inverse_normalised_rank


