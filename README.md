##  Weakly Supervised Detection of Marine Animals in High Resolution Aerial Images 

This repository contains the code and resources for the article " Weakly Supervised Detection of Marine Animals in High Resolution Aerial Images ".

<img width="800" alt="graphical_abstract" src="https://user-images.githubusercontent.com/9824244/155295184-90b9c06b-3bd8-4af6-bbb1-fd16f6f2acd9.png">

### Dependencies

Developed using Python version `3.7.1`.
```
pytorch>=1.8
torchvision>=0.8.1
keepsake>=0.4.2
opencv>=4.1.2
```

### Architecture

Clone using the following command:

```
git clone --recurse-submodules https://github.com/Pangoraw/MarineMammalsDetection
```

The folder structure is the following:

```
└──code
   ├── examples # scripts consuming the padim library for training + testing
   │   ├── main.py # main entry point
   │   ├── mvtec.py # training mvtec dataset
   │   ├── padeep.py # training deep methods for semmacape/kelonia/ifremer datasets
   │   ├── semmacape.py # training regulart methods for semmacape/kelonia/ifremer
   │   ├── test_mvtec.py # testing pipeline for the mvtec ad dataset
   │   ├── test_semmacape.py # testing pipeline for the semmacape/kelonia/ifremer dataset
   │   └── train_test.sh # bash script to launch main.py on the slurm cluster with a config
   ├── padim # main library code and implementations of the models
   │   ├── backbones # various encoding backbones
   │   ├── base.py # common encoding for all implementations
   │   ├── datasets # torch utils datasets for the project
   │   ├── deep_svdd # deep-svdd implementation of a MLP
   │   ├── multi_svdd.py # multi-headed deep-svdd
   │   ├── padim.py # regular padim
   │   ├── padim_shared.py # padim with a single shared Gaussian
   │   ├── padim_svdd.py # padim with a multi headed deep-svdd
   │   ├── panf.py # padim with a multi headed normalizing flow
   │   └── utils
   │       ├── utils.py # various utilities from the great https://github.com/taikiinoue45/PaDiM 
   │       ├── distance.py # mahalanobis distance GPU implementation
   │       └── regions.py # region proposals and utilities
   └── configs # pre-defined configs to use with examples/train_test.sh
```

### Main script (`examples/main.py`)

The main entry point is `examples/main.py` which is going to train + test the specified model. If the file at location `params_path` already exists, the script is going to load it and only perform testing for reproducibility.

| Parameter | Description |
| --------- | ----------- |
|`train_folder TRAIN_FOLDER`|Training image folder location|
|`test_folder TEST_FOLDER`|Testing image folder location|
|`params_path PARAMS_PATH`|Models parameters saving path|
|`train_limit TRAIN_LIMIT`|# of training samples (optional)|
|`load_path LOAD_PATH`|Encoder weights param file (optional)|
|`threshold THRESHOLD`|Patch anomaly threshold (optional)|
|`test_limit TEST_LIMIT`|# of testing samples (optional)|
|`iou_threshold IOU_THRESHOLD`|IoU threshold for positive predicitions|
|`min_area MIN_AREA`|Minimum area for boxes, smaller ones are filtered out (optional)|
|`use_nms`|Use non-maximum suppression (optional)|
|`shared`|Train a single shared Gaussian estimator (optional)|
|`deep`|Use a "deep" method like PaDiM+Deep-SVDD or PaDiM+NF (optional)|
|`semi_ortho`|Use semi-orthogonal encoding instead of random dimensions selection (optional)|
|`compare_all`|Use cross Mahalanobis distance between all patches and all Gaussian estimators|
|`size SIZE  [default=416x416]`|Input image size|
|`oe_folder OE_FOLDER`|Outlier images folder (optional)|
|`oe_frequency OE_FREQUENCY`|Outlier image frequency (2 = 50% of images are outliers, 3 = 1/3,etc...)|
|`n_epochs N_EPOCHS`|Number of training epochs for deep methods|
|`ae_n_epochs AE_N_EPOCHS`|Number of pre-training epochs for Deep-SVDD|
|`n_svdds N_SVDDS`|Number of heads for deep methods|
|`pretrain`|Use pretraining for Deep-SVDD|
|`use_self_supervision`|Use self supervised loss|
|`num_embeddings NUM_EMBEDDINGS`|Size of embedding vectors|
|`backbone {resnet18,resnet50,wide_resnet50}`|Which encoding backbone to use|
|`nf`|Whether to use Deep-SVDD or NF for deep methods|

### Citing

If this work is useful to your research, consider citing the following article:

```bib
@Article{rs14020339,
	AUTHOR = {Berg, Paul and Santana Maia, Deise and Pham, Minh-Tan and Lefèvre, Sébastien},
	TITLE = {Weakly Supervised Detection of Marine Animals in High Resolution Aerial Images},
	JOURNAL = {Remote Sensing},
	VOLUME = {14},
	YEAR = {2022},
	NUMBER = {2},
	ARTICLE-NUMBER = {339},
	URL = {https://www.mdpi.com/2072-4292/14/2/339},
	ISSN = {2072-4292},
	DOI = {10.3390/rs14020339}
}
```
