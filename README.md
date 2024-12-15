# Enriching operational high-resolution ensemble forecasts with StyleGAN
A new proposal to generate ensemble forecasts enriching the AROME Ensemble Prediction System. 
The Stylegan-2 network is a well-performing GAN architecture (see the [original implementation](https://github.com/NVlabs/stylegan2) and the [pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch). The goal is to use a pretrained Stylegan-2 generator in order to enrich AROME-EPS forecasts. We do so by using the W hidden representation and existing AROME members to generate new members at a reduced numerical cost.  

The code allows to *train a GAN from scratch*, then to use *inversion* to retrieve important information from "real" (classically computed) AROME ensembles, and *perturbation* to generate new, randomly perturbed ensembles. Additionally, a *tuning* procedure allows to control the intensity of the perturbation.

Most of the core code for GAN training is taken as is from [Rosinality's stylegan2-pytorch github page](https://github.com/rosinality/stylegan2-pytorch) and adapted to run on Meteo France clusters. 

The related paper can be found here: ??

Contributors : C. Brochet, G. Moldovan, L. Raynaud, B. Gandon, L. Poulain--Auzéau, J. Rabault, C. Regan, V.Sanchez.

# Installing

Run a conda / docker environment with pytorch==2.1.0.
Then make sure `ninja` is installed (in order to run optimized kernels of stylegan-2):

```
apt update
apt install -y curl ninja-build build-essential
```

Finally : 

`pip install -r requirements.txt`

In order to run the accompanying scripts, you must define 2 environment variables :

`STYLEGAN_CODE_DIR` :  the location of your code, ending with `enriching_gan_ensembles` (the name of the repo).

`STYLEGAN_DATA_DIR` : the location of your data's _parent directory_ (e.g if your raw data samples are in `/disk/directory/enriching_data/data/`, then please provide `/disk/directory`).

# Running scripts

Scripts must be launched from the repo's main directory.

## Training
`bash scripts/run_training.sh $N_GPUs`
with $N_GPUs the number of graphic card you want to train on. This code has been tested with *single-node* training, larger scale training may encur communication overheads.
## Inversion
`bash scripts/run_inversion.sh`
## Perturbation tuning
`bash scripts/scale_tune.sh`
## Ensemble enriching
`bash scripts/generate_condition.sh`

# Data
## The AROME-EPS Dataset

The whole dataset comprises 516 AROME ensemble forecasts covering the period from June 15th, 2020, to November 12th, 2021. Each ensemble forecast is composed of 16 members and includes lead times at 1-hour intervals, ranging up to 45 hours. It follows that [516x45x16=371520]() individual samples are available for training if each members of the enseble at a given lead time is considered individually.

The data is restricted to a region encompassing the South-Eastern quarter of metropolitan France with a resolution of [256x256]. Three variables are here considered: the horizontal (u) and vertical (v) components of the wind speed vector at 10 meters and the temperature at 2 meters (t2m). Each individual sample can be conceptualized as a tensor with 3 channels, a width of 256 and a height of 256 [3, 256, 256]. Hourly precipitation rates (rr) are a 4th variable, which can be dealt with as an option (results are not validated for this latter variable).

To efficiently load and organize the dataset, a metadata CSV file is utilized. The file structure is illustrated below:

| Name          | Importance | PosX | PosY | Date       | LeadTime | Member |
|---------------|------------|------|------|------------|----------|--------|
| ...           | ...        | ...  | ...  | ...        | ...      | ...    |
| _sample1440   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 0      |
| _sample1441   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 1      |
| _sample1442   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 2      |
| _sample1443   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 3      |
| _sample1444   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 4      |
| _sample1445   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 5      |
| _sample1446   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 6      |
| _sample1447   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 7      |
| _sample1448   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 8      |
| _sample1449   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 9      |
| _sample1450   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 10     |
| _sample1451   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 11     |
| _sample1452   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 12     |
| _sample1453   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 13     |
| _sample1454   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 14     |
| _sample1455   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 15     |
| _sample1456   | 1,0        | 256  | 256  | 2021-06-02 | 1        | 0      |
| _sample1457   | 1,0        | 256  | 256  | 2021-06-02 | 1        | 1      |
| _sample1458   | 1,0        | 256  | 256  | 2021-06-02 | 1        | 2      |
| ...           | ...        | ...  | ...  | ...        | ...      | ...    |

- **`Name`**: A unique identifier for each sample.
- **`Importance`**: Importance level.
- **`PosX` and `PosY`**: Size of the image
- **`Date`**: Date of the ensemble forecast.
- **`LeadTime`**: Lead time in hours.
- **`Member`**: Member index within the ensemble.

This metadata file plays a crucial role in loading the dataset efficiently and ensuring the proper association of each sample with its corresponding attributes. Please update the file path in your code to reflect the location of your metadata CSV file.

# Configurations:


## TRAINING

- **data_dir**: Path to the directory containing input data.
- **output_dir**: Path to the directory where the experiment outputs will be stored.
- **config_dir**: Path to the directory containing configuration files. Example:
- **id_file**: Relative path to the CSV file containing labels, relative to data_dir.

#### Experiment Parameters (ensemble)

**General Parameters:**

- **total_steps**: Number of total steps for each experiment. Example: [500001]
- **epochs_num**: Number of epochs for each experiment. Example: [25]
- **pretrained_model**: Step indices for pretrained models. Set to -1 if no pretrained model. Example: [108000]

**Generator and Discriminator Configuration:**

- **var_names**: Variable names for used training. Example: ["[rr,u,v,t2m]"]
- **batch_size**: Real batch size is batch_size * N, where N is the number of GPUs. Example: [8]
- **lr_D**: Learning rate for the discriminator. Example: [0.002]
- **lr_G**: Learning rate for the generator. Example: [0.002]
- **g_channels**: Number of generator channels. Example: [3]
- **d_channels**: Number of discriminator channels. Example: [3]
- **path_batch_shrink**: Batch shrinkage factors for the path regularization. Example: [2]
- **tanh_output**: Use of tanh output. Example: [True]

**StyleGAN Configuration:**

- **model**: Model name. Example: ['stylegan2']
- **train_type**: Training type. Example: ['stylegan']
- **latent_dim**: Dimension of the latent space. Example: [512]
- **use_noise**: Use of noise injection. Example: [False]

**Database Parameters:**

- **crop_indexes**: Crop indexes for input AROME forecasts. Example: ["[0,256,0,256]"]
- **crop_size**: Crop size. Example: ["[256,256]"]
- **full_size**: Full size of input AROME forecasts. Example: ["[256,256]"]

**Configuration Files for dataset handling and scheduler config:**

- **dataset_handler_config**: Relative paths to dataset handler configuration files. Example: ['dataset_handler_config.yaml']
- **scheduler_config**: Relative paths to scheduler configuration files. Example: ['scheduler_config.yaml']


### dataset_handler_config.yaml

The configuration file includes various settings for preprocessing and normalization of the precipitation variable. Below are the details of each parameter:

- **stat_version**: Name of the statistical file. The actual file name would be, for example, `mean_[stat_version]_log_ppx.npy`. Example: '"rr"'

#### With precipitation variables rr (For_rr)
- **log_transform_iteration**: Number of times the log transformation is applied to the variable 'rr'. Example: '1'

- **symetrization**: Whether symmetrization is applied to the variable 'rr'. Example: 'False'

- **gaussian_std**: Threshold between rain and no rain to add Gaussian noise where 'rr < gaussian_std'. Example: '0'

#### Normalization

- **type**: Type of normalization to be applied. Choose between '"mean"', '"minmax"', or '"None"'. Example: '"minmax"'

- **per_pixel**: Whether normalization is applied with global values to each pixel or specific pixel values. Example: 'False'

    If `per_pixel` is `True`, the following options are used. For the 'rr' variable:

    - **blur_iteration**: The number of times a Gaussian convolution is applied to the grid containing the max/min/mean/max_std. Example: '1'


## INVERSION TO THE LATENT SPACE

Once a skilled Generator is obtained, one can invert real AROME ensemble forecasts to the latent space using [main_inversion](https://github.com/flyIchtus/styleganPNRIA/blob/main/main_inversion.py). The inversion is configurated with the following parser parameters:

#### Directory Paths

- **`--ckpt_dir`**: Path to the checkpoint directory containing the pre-trained StyleGAN model.  
  - *Default*: ``

- **`--real_data_dir`**: Path to the directory containing real data used for inversion.  
  - *Default*: ``

- **`--output_dir`**: Path to the directory where the inversion results will be stored.  
  - *Default*: ``

- **`--pack_dir`**: Path to the directory where the real normalized ensembles that are inverted are stored.  
  - *Default*: ``

- **`--mean_file`**: File containing mean values for normalization.  
  - *Default*: ``

- **`--max_file`**: File containing max values for normalization.  
  - *Default*: ``

- **`--device`**: Device to run the inversion on (e.g., 'cuda:0').  
  - *Default*: `'cuda:0'`

#### Inversion Parameters. For more details, check the original [StyleGAN2 paper](https://arxiv.org/abs/1912.04958) and the [implementation](https://github.com/rosinality/stylegan2-pytorch) this repository is based on.

- **`--lr_rampup`**: Duration of the learning rate warmup.  
  - *Default*: `0.05`

- **`--lr_rampdown`**: Duration of the learning rate decay.  
  - *Default*: `0.25`

- **`--lr`**: Learning rate for optimization.  
  - *Default*: `0.1`

- **`--noise`**: Strength of the noise level.  
  - *Default*: `0.005`

- **`--noise_ramp`**: Duration of the noise level decay.  
  - *Default*: `0.75`

- **`--invstep`**: Number of optimization iterations.  
  - *Default*: `1000`

- **`--var_indices`**: List of variable indices to invert (e.g., [1,2,3]). Highly dependant on the shape of the samples of the dataset.  
  - *Default*: `[1,2,3]`

- **`--Shape`**: Size of the samples as a tuple (channels, height, width).  
  - *Default*: `(3,256,256)`

- **`--noise_regularize`**: Weight of the noise regularization during inversion.  
  - *Default*: `10e5`

- **`--loss`**: Type of loss function used (options: 'mse' or 'mae').  
  - *Default*: `'mse'`

- **`--loss_intens`**: Weight of the pixel loss.  
  - *Default*: `1.0`

- **`--inv_checkpoints`**: List of optimization steps to save results.  
  - *Default*: `[200,400,600,800,1000]`

#### Data Control for Inversion

- **`--dates_file`**: CSV file containing dates for inversion.  
  - *Default*: `'Large_lt_test_labels.csv'`

- **`--date_start`**: Start date for inversion in the format 'YYYY-MM-DD'.  
  - *Default*: `'2021-06-01'`

- **`--date_stop`**: Stop date for inversion in the format 'YYYY-MM-DD'.  
  - *Default*: `'2021-15-11'`

- **`--leadtimes`**: List of lead times for inversion.  
  - *Default*: `[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]`

## GENERATION OF GAN-ENRICHED ENSEMBLES

Once a skilled Generator is obtained real ensemble members have successfully been inverted to the latent space, one can enrich this ensembles using [main_perturbation.py](https://github.com/flyIchtus/styleganPNRIA/blob/main/main_perturbation.py)

#### Directory Paths

- **`--ckpt_dir`**: Path to the directory containing the pre-trained StyleGAN checkpoint.  
  - *Default*: `''`

- **`--real_data_dir`**: Path to the directory containing the full dataset.  
  - *Default*: `''`

- **`--data_dir`**: Path to the data directory containing the inversed ensembles.  
  - *Default*: `''`

- **`--output_dir`**: Path to the directory where the gan-enriched ensembles will be stored.  
  - *Default*: `''`

- **`--pack_dir`**: Path to the directory where the normalized real ensembles are stored.  
  - *Default*: `''`

- **`--mean_file`**: File containing mean values for normalization.  
  - *Default*: `''`

- **`--max_file`**: File containing max values for normalization.  
  - *Default*: `''`

- **`--var_indices`**: List of variable indices to be used.  
  - *Default*: `[1,2,3]`

- **`--Shape`**: Size of the samples as a tuple (channels, height, width).  
  - *Default*: `(3,256,256)`

- **`--N_samples`**: Ensemble size of the generated ensembles.  
  - *Default*: `120`

- **`--inv_step`**: Which step of the inversion process of the real ensembles should be used.  
  - *Default*: `1000`

#### Perturbation Parameters

- **`--sample_rule`**: Perturbation method used for generating new ensembles (options: 'random', 'normal', 'w', 'extrapolation').  
  - *Default*: `'random'`

- **`--style_indices`**: Which vectors of the latent code should be perturbed.  
  - *Default*: `'[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'`

- **`--conditioning_members`**: Number of members used to generate perturbed ensembles (Max = 16).
  - *Default*: `16`

## Data Control for Perturbation

- **`--dates_file`**: CSV file containing dates for the ensembles to be enriched.  
  - *Default*: `''`

- **`--date_start`**: Start date for the ensembles to be enriched in the format 'YYYY-MM-DD'.  
  - *Default*: `'2020-07-01'`

- **`--date_stop`**: Stop date for the ensembles to be enriched in the format 'YYYY-MM-DD'.  
  - *Default*: `'2020-12-31'`

- **`--leadtimes`**: List of lead times for perturbation or inversion.  
  - *Default*: `[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]`
