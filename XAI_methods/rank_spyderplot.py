import pandas as pd    
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
from get_spyderplot import spyder_plot

rootdir = "XAI_methods/Saved_Ranks"

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        
        if file == "Avg metric scores.csv":
            path = '{}/{}'.format(subdir, file)
            df = pd.read_csv(path)
            eps = 10e-6
            df = df.set_index('Unnamed: 0')
            df = df.apply(pd.to_numeric, errors='coerce')
            eps = 1e-8
            df_normalised = df.loc[:, ~df.columns.isin(['Robustness', 
                                'Randomization'])].apply(lambda x: x / (x.max() + eps))
            
            df_normalised["Robustness"] = df["Robustness"].min()/(df["Robustness"].values + eps)
            df_normalised["Randomisation"] = df["Randomization"].min()/(df["Randomization"].values + eps)
            df_inverse = df_normalised.copy()  # Create a copy of the normalized DataFrame

            # Negative normalization for plotting
            df_inverse = df.loc[:, ~df.columns.isin(['Robustness', 
                                'Randomization'])].apply(lambda x: (x.max() - x) / (x.max() - x.min() + eps))

            # Reverse normalization for 'Robustness'
            df_inverse['Robustness'] = df['Robustness'].min() / (df_normalised['Robustness'].values + eps)

            # Reverse normalization for 'Randomisation'
            df_inverse['Randomisation'] = df['Randomization'].min() / (
                                df_normalised['Randomisation'].values + eps)
            df_normalised_rank = df_inverse.rank(ascending = False)
            xai_methods = {
            "Lime": None,
            "HiResCAM": None,
            "GradientShap": None,
            "Saliency": None,
            "Occlusion": None
        }
            df_normalised_rank.to_csv((path + 'Average rank per model.csv'))
            data = [df_normalised_rank.columns.values, (df_normalised_rank.to_numpy())]
            theta = spyder_plot(len(data[0]), frame='polygon')
            spoke_labels = data.pop(0)
            colours_order = ["red", "darkorange", "royalblue", "darkgreen", "slateblue", "purple"]
            include_titles, include_legend = False, False


            fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(top=0.85, bottom=0.05)
            for i, (d, method) in enumerate(zip(data[0], xai_methods)):
                line = ax.plot(theta, d, label=method, color=colours_order[i], linewidth=5.0)
                ax.fill(theta, d, alpha=0.15)



            # Set lables.
            if include_titles:
                ax.set_varlabels(labels=["Faithfulness", "Localization", 
                                         "Complexity", "Axiomatic","Robustness","Randomization"])
            else:
                ax.set_varlabels(labels=[]) 
            ax.set_rgrids(np.arange(0, df_normalised_rank.values.max() + 0.5), labels=[]) 

            # Put a legend to the right of the current axis.
            if include_legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))


            fig.tight_layout()
            fig.savefig(path + "spyderplot_metrics.png", dpi=fig.dpi)
            plt.close()