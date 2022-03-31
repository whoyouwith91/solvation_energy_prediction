<h1 align="center">Accurate prediction of aqueous free solvation energies using 3D atomic feature-based graph neural network with transfer learning</h1>
<h4 align="center">Dongdong Zhang, Song Xia, and Yingkai Zhang</h4>

![model architecture](model.jpg)

The repository contains all of the code and instructions needed to reproduce the experiments and results of **[Accurate prediction of aqueous free solvation energies using 3D atomic feature-based graph neural network with transfer learning]**. We show the whole process from datasets to model training step-by-step.

## 1. Datasets downloading
Since the 3D structures are stored in SDF and XYZ formats for Frag20-Aqsol-100K, they are saved elsewhere and can be downloaded either from our IMA website or using the following command line. 
- To download and save SDF files for MMFF-optimized geometries: 
> `cd ./data/Frag20-Aqsol-100K/sdf/MMFF/`  (navigate to the corresponding directory)  
> `wget link` (link can be copied from IMA)  
- To download XYZ files for MMFF-optimized geometries:   
> `cd ./data/Frag20-Aqsol-100K/xyz/MMFF/`  (navigate to the corresponding directory)  
> `wget link` (link can be copied from IMA)  
- To download SDF files for QM-optimized geometries: `wget link`
- To download XYZ files for QM-optimized geometries: `wget link`
After 
## 2. Data preprocessing

## 3. Training