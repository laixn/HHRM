# HHRM: a hierarchical recursive model for hierarchical molecular subtype classification in pan-cancer
[![Static Badge](https://img.shields.io/badge/Preprint-bioRxiv-darkred)]([https://doi.org/](https://doi.org/10.64898/2026.10.30.761789))
[![Static Badge](https://img.shields.io/badge/ZENODO-10.5281/zenodo.23589666-red)](https://zenodo.org/uploads/23589666)

We present a cancer hallmark-based hierarchical recursive model (HHRM) for hierarchical multi-label classification in pan-cancer. HHRM integrates hierarchical label structures with recursive neural networks, enabling simultaneous modeling of multi-level dependencies from broad cancer categories to specific subtypes. We trained the model using a comprehensive, pan-cancer cohort of 11,476 patients with 26 primary cancers, with subtypes spanning up to eight levels. HHRM achieved accuracy of 0.7924 for predicting hierarchical tumor subtypes. 

<p align="center">
  <img width="762" height="537" alt="Image" src="https://github.com/user-attachments/assets/7530997b-e04d-4996-a16c-fe5ffcf18330" />
</p>

## The current version is to provide reviewers with reproducible experimental results
The repository contains the following strucutre and files:
```bash
main/
  └──code/ # the code to run the model
        ├── HHRM_training_up.py
        └── load_and_pred_up.py
  └──data/ # the data used to train and test the model
        ├── cleaned_data_out.csv
        └── cleaned_label.xlsx
  └──adjacency_matrix/ # the cancer hallmark matrix for GCN
        ├── Undirected_0...matrix.npz
        ├── ...
        └── Undirected_9...matrix.npz
```

## Pre-requisites: TEST 
* Linux (Ubuntu 20.04.6 LTS) 
* NVIDIA GPU (NVIDIA GeForce RTX 3090 24G)
* Python (3.8.20), Pytorch (2.3.0), Numpy (1.24.3), Pandas (2.0.3), scikit-learn (1.3.0), SHAP (0.41.0)    

## Run the code:
1. You first need to download the archived data and models from [![Static Badge](https://img.shields.io/badge/ZENODO-grey)](https://zenodo.org/uploads/23589666)
2. You must ensure that the input file path required in the code file is correct.
3. Run the file `code/HHRM_training_up.py` for training, and run the file `code/load_and_pred_up.py` for load the best model and predict analysis.
4. If you want to calculate the SHAP values, please set `Whether_to_calculate_the_shap = TRUE` in `code/load_and_pred_up.py`.
5. If you want to predict the hard samples, please set `Whether_to_predict_hard_samples = TRUE` in `code/load_and_pred_up.py`.

## Usage & Citation 
If you find our work useful, please consider citing it:

Qingsong Zhang, Fei Liu, Xin Lai. HHRM test 
Bioinformatics-test. https-test.

```bash
@article{Zhang_HHRM_2026,
  title={Classifying tumors from broad category to hierarchical molecular subtype: an interpretable hierarchical framework for precision oncology},
  author={Qingsong Zhang, Fei Liu, Xin Lai},
  journal={bioRxiv},
  doi={[10.64898/2026.10.30.761789](https://doi.org/10.64898/2026.10.30.761789 )},
  year={2026}
}
```


©[Lai Lab](https://sites.google.com/view/lai-lab) - This code is available under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0) license.
```
