<h1 align="center">Accurate prediction of aqueous free solvation energies using 3D atomic feature-based graph neural network with transfer learning</h1>
<h4 align="center">Dongdong Zhang, Song Xia, and Yingkai Zhang</h4>

![model architecture](model.jpg)

The repository contains all of the code and instructions needed to reproduce the experiments and results of **[Accurate prediction of aqueous free solvation energies using 3D atomic feature-based graph neural network with transfer learning](https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00260)**. We show the whole process from datasets to model training step-by-step.

## Table of Contents
- Project organization
- Conda environment setup  
- Datasets downloading   
- Data preprocessing  
- Training
- References
---

## 1. Project organization
```
|- README.md                                <- this file
|- scr                                      <- Main code of this work
    |- train.py                             <- model training, see usage in Step 5
    |- trainer.py                           <- provide train/test functions
    |- model.py                             <- model building
    |- gnns.py                              <- a list of GNN modules
    |- k_gnn.py                             <- dataloader building
    |- supergat_conv.py                     <- SuperGAT 
    |- layers.py                            <- simple layers building blocks
    |- prepare_data.py                      <- process molecules datasets
    |- featurization.py                     <- featurization methods
    |- helper.py                            <- provide simple functions for use
    |- args.py                              <- list of argments for training
|- data                                     <- where data are saved       
    |- Frag20-Aqsol-100K
        |- split                            <- train/validation/test csv files    
    |- FreeSolv
        |- split                            <- train/validation/test csv files    
        |- sdf                              <- SDF files for all molecules   
        |- xyz                              <- XYZ files for all molecules   
|- models                                   <- where pretrained models are saved for use
    |- Frag20-Aqsol-100K
        |- pretrained
            |- 3D_MMFF                      <- model trained with A3D features calculated from MMFF-opt geometries 
            |- 3D_QM                        <- model trained with A3D features calculated from QM-opt geometries 
            |- 2D                           <- model trained with 2D features 
|- results                                  <- where training results should be saved 
```

## 2. Conda environment setup: 
Python 3.8 is recommended here with the **[miniconda3](https://docs.conda.io/en/latest/miniconda.html)**. 
The package installation order is recommended as below: 
- PyTorch.   
`conda install pytorch cudatoolkit=10.2 -c pytorch`  
To be noted, in order to be compatible with the installation of Torch-geometric, cuda10.2 for torch should be used here. 
- **[Torch geometirc](https://github.com/pyg-team/pytorch_geometric)**.  

`conda install pyg -c pyg`

- **[rdkit](https://www.rdkit.org/docs/Install.html)**.  

`conda install -c conda-forge rdkit`

- **[DSCRIBE](https://singroup.github.io/dscribe/latest/install.html)**.  

`conda install -c conda-forge dscribe`

- PrettyTable.  

`conda install -c conda-forge prettytable`

## 3. Datasets downloading
Since the 3D structures are stored in SDF and XYZ formats for Frag20-Aqsol-100K, they are saved elsewhere and can be downloaded either from our IMA website or using the following command line. 

`cd data`

`wget PLACE_HOLDER`

After downloading the tar.bz2 file, unzip it using `tar -xf`. 

`tar xvf Frag20-Aqsol-100K.tar.bz2`

You should see a list of folders. Then check the total number of files: 

`find Frag20-Aqsol-100K/ -type f | wc -l` 

which should return around 400,000. Now return to the root folder:

`cd ..`


## 4. Data preprocessing
To generate molecule graph datasets for Torch geometric reading, the `preprae_data.py` contains the codes for Frag20-Aqsol-100K and FreeSolv. For example, the following command line is used to process each molecule in Frag20-Aqsol-100K by featurizing atoms/bonds using 3D atomic features.   
`cd src`

`python prepare_data.py  --data_path ../data --save_path ../data/processed --dataset Frag20-Aqsol-100K --ACSF --cutoff 6.0 --xyz MMFF --train_type TS --tqdm`  

`python prepare_data.py  --data_path ../data --save_path ../data/processed --dataset FreeSolv --ACSF --cutoff 6.0 --xyz MMFF --train_type FT --tqdm` 


where `--data_path` is the path to the place where folders sdf, xyz and split are all saved, `--save_path` is the same path as `data_path` for default where a new directory `graphs` will be created. `--ACSF` means the 3D features are used here. `--cutoff` is the parameter for ACSF functions. `--xyz` means if the MMFF-optimized or QM-optimized geometries are used. `--train_type` means if we save the graph datasets to train the model from scratch (TS) or not. 

After running the above script, there will be a directory generated in the `--save_path` in the format of `graphs/3D_TS_MMFF/raw` where 3D means `ACSF` is called here, MMFF means `xyz` is using MMFF geometries, TS means the `train_type` is training from scratch (TS), and `raw` is the required folder by torch geometric. 

## 5. Training
To train different tasks, the `train.py` contains the codes. For example, the following command line is used to process each molecule in Frag20-Aqsol-100K by featurizing atoms/bonds using 3D atomic features.  

`python train.py --data_path ../data/processed --running_path ../results --dataset Frag20-Aqsol-100K --gnn_type pnaconv --seed 111 --train_type TS --style 3D_FT_MMFF --experiment frag20-100k-ts --fully_connected_layer_sizes 120 120 --bn --residual_connect --data_seed 456 --train_size 80000 --val_size 10000 --test_size 10000`


where `--data_path` is same as the above `--save_path` in Step 3, `--results_path` is where the results are saved. `--gnn_type` is the GNN module used to build the model. For example, `pnaconv`. `--seed` is the random seed to initialize the weights. `--style` should be consistent with the folder where processed datasets are saved, for example, `3D_TS_MMFF` from the script in Step 3. `--experiment` is any experiment names for the job running. `[120 60]` means the dimensions for readout layers are 120, 60, except for the last layer with 1 be the dimension. `--bn` means using the batch normalization. `--residual_connect` means using the residual connection. `--data_seed` only works when `--sample` is called to do the random splits. 

After running the script, there will be a new directory generated in the format of `--running_path` + `Frag20-Aqsol-100K` + `1-GNN` + `--gnn_type` + `--experiment`, where file, `config.json`, training results file `data.txt`, and two folders where the model parameters for best model (lowest validation error) and last-epoch model are saved: `model_best.pt` and `last_model.pt`. 

If finetuning the FreeSolv task, argment `--preTrainedPath` should be explicitly called. It should point to the direcotry where `model_best.pt` is saved. Here in this repo, we saved our previously trained models on Frag20-Aqsol-100K using MMFF-optimized geometries, QM-optimized geometris and 2D for use, which are saved in `models/Frag20-Aqsol-100K/pretrained/`. 

`python train.py --data_path ../data/processed --running_path ../results --dataset FreeSolv --gnn_type pnaconv --seed 222 --train_type FT --style 3D_FT_MMFF --experiment freesolv-ft --fully_connected_layer_sizes 120 120 --bn --residual_connect --data_seed 123 --train_size 504 --val_size 63 --test_size 63 --optimizer adam --preTrainedPath ../models/Frag20-Aqsol-100K/pretrained/3D_MMFF/`

## 6. References
- ChemProp: https://github.com/chemprop/chemprop  
- k-gnn: https://github.com/chrsmrrs/k-gnn  
- SuperGAT: https://github.com/dongkwan-kim/SuperGAT  
