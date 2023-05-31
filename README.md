# Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification:
**Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification**

_Akshatha Mohan and Joshua Peeples_

Note: If this code is used, cite it: Akshatha Mohan and Joshua Peeples. 
(2023, July 19). Peeples-Lab/XAI_Analysis: Initial Release (Version v1.0). 
Zendo. https://doi.org/10.5281/zenodo.5572704 (TBD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5572704.svg)](https://doi.org/10.5281/zenodo.5572704)

[[`IEEE Xplore (IGARRS, TBD)`](https://ieeexplore.ieee.org/document/9706894)]

[[`arXiv, TBD`](https://arxiv.org/abs/2110.05324)]

[[` BibTeX`](#CitingQuantitative)]

In this repository, we provide the paper and code for the "Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification."

## Installation Prerequisites

This code uses python, pytorch, quantus, and captum. 
Please use [[`Pytorch's website`](https://pytorch.org/get-started/locally/)] to download necessary packages.
[Quantus](https://quantus.readthedocs.io/en/latest/getting_started/installation.html) 
and [Captum](https://captum.ai/#quickstart) are used for the XAI model. Please follow the instructions on each website to download the modules.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. 

## Main Functions

The XAI Analyis runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(**Parameters)```

2. Prepare dataset(s) for model

 ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/Peeples-Lab/XAI_Analysis

└── root dir
    ├── demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameters file for demo.
    ├── Prepare_Data.py  // Load data for demo file.
    └── Utils  //utility functions 
        ├── Generating_Learning_Curves.py  // Plot training and validation accuracy and error measures.
        ├── Generate_TSNE_visual.py  // Create TSNE visual for results.
        ├── Network_functions.py  // Contains functions to initialize, train, and test model. 
        ├── pytorchtools.py // Function for early stopping.
        ├── Save_Results.py  // Save results from demo script.
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) 
file in the root directory of this source tree.

This product is Copyright (c) 2023 A. Mohan and J. Peeples. All rights reserved.

## <a name="CitingQuantitative"></a>Citing LACE

If you use the code, please cite the following 
reference using the following entry.

**Plain Text:**

A. Mohan and J. Peeples, "Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification,"  
In 2023 IEEE International Geoscience and Remote Sensing Symposium IGARSS, 
pp. TBD. IEEE, 2023

**BibTex:**
```
@inproceedings{mohan2023quantitative,
  title={Explainable Systematic Analysis for Synthetic Aperture Sonar Imagery},
  author={Mohan, Akshatha and Peeples, Joshua},
  booktitle={2023 IEEE International Geoscience and Remote Sensing Symposium IGARSS},
  pages={TBD},
  year={2023},
  organization={IEEE}
}

```
