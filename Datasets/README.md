# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. Please follow the following instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

## UCMerced dataset [[`BibTeX`](https://github.com/Peeples-Lab/XAI_Analysis/tree/main/Datasets#citing-ucmerced)]
To download the UCMerced dataset, initialize a new UC Merced dataset instance in which download is set to True.
The structure of the `UCMerced` folder is as follows:
```
└── root dir
    ├── UCMerced_LandUse //Contains folders of images for each class.
    ├── uc_merced-train
    ├── uc_merced-test  
    ├── uc_merced-val
``` 

## <a name="CitingUCMerced"></a>Citing UCMerced
If you use this dataset in your research, please cite the following paper:

https://dl.acm.org/doi/10.1145/1869790.1869829

**BibTex:**
```
@inproceedings{yang2010bag,
  title={Bag-of-visual-words and spatial extensions for land-use classification},
  author={Yang, Yi and Newsam, Shawn},
  booktitle={Proceedings of the 18th SIGSPATIAL international conference on advances in geographic information systems},
  pages={270--279},
  year={2010}
}
```

## EuroSAT dataset [[`BibTeX`](https://github.com/Peeples-Lab/XAI_Analysis/tree/main/Datasets#citing-eurosat)]
To download the EuroSAT dataset, initialize a new EuroSAT dataset instance in which download is set to True.
The structure of the `EuroSAT` folder is as follows:
```
└── root dir
    ├── ds //Contains folders of images for each class.
    ├── eurosat-train
    ├── eurosat-test
    ├── eurosat-val
``` 

## <a name="CitingEuroSAT"></a>Citing EuroSAT
If you use this dataset in your research, please cite the following paper:
https://ieeexplore.ieee.org/document/8736785

**BibTex:**
```
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={12},
  number={7},
  pages={2217--2226},
  year={2019},
  publisher={IEEE}
}
```

## MSTAR dataset [[`BibTeX`](https://github.com/Peeples-Lab/XAI_Analysis/tree/main/Datasets#citing-mstar)]
To download the MSTAR dataset. Download the <a href="https://github.com/jangsoopark/AConvNet-pytorch/releases/download/v2.2.0/dataset.zip">download.zip</a>.
Extract the eoc-1-t72-a64 folder from it.
The structure of the `MSTAR` folder is as follows:
```
└── root dir
    └── eoc-1-t72-a64 //Contains folders of images for each class.
	├── train
	├── test
``` 

## <a name="CitingMSTAR"></a>Citing MSTAR
If you use this dataset in your research, please cite the following paper:
<a href="https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2757/0000/MSTAR-extended-operating-conditions-a-tutorial/10.1117/12.242059.full?SSO=1">MSTAR extended operating conditions: a tutorial </a>


**BibTex:**
```
@article{keydel1996mstar,
  title={MSTAR extended operating conditions: A tutorial},
  author={Keydel, Eric R and Lee, Shung Wu and Moore, John T},
  journal={Algorithms for Synthetic Aperture Radar Imagery III},
  volume={2757},
  pages={228--242},
  year={1996},
  publisher={SPIE}
}
```
