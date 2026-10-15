# HHRM
HHRM: a biologically informed classifier for hierarchical molecular subtype classification in pan-cancer

We present a hierarchical model, HHRM, to classify hierarchical tumor subtypes in human cancer.


<p align="center">
  <img src="https://github.com/user-attachments/assets/e0969ea7-38e6-4332-bdf2-892bdb75f3b0" alt="居中图片" style="max-width: 100%;">
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
        └── cleaned_data_test.xlsx
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
1. test
2. test

## Usage & Citation 
If you find our work useful, please consider citing it:

Qingsong Zhang, Fei Liu, Xin Lai. HHRM test 
Bioinformatics-test. https-test.

```bash
@article{Zhang_HHRM_2026,
  title={HHRM-test},
  author={Qingsong Zhang, Fei Liu, Xin Lai},
  journal={test},
  doi={test},
  year={2026}
}
```


©[Lai Lab](https://sites.google.com/view/lai-lab) - This code is available under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0) license.
```
