# [Distribution-Aware Tensor Decomposition for Compression of Convolutional Neural Networks](https://arxiv.org/abs/2511.04494)

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-4b44ce.svg)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.04494-b31b1b.svg)](https://arxiv.org/abs/2511.04494)

## Computation of Sigma matrix and Tucker-2 Decomposition via ALS-Sigma

This repository contains computation of Sigma matrix using the Cholesky decomposition and implementation of the Tucker decomposition algorithm using Alternating Least Squares (ALS) with respect to Sigma norm. 

## Overview
Consider the Sigma matrix $\Sigma := \mathbb{E}[xx^T]$. Firstly, we compute the square root of the Sigma matrix $\Sigma^{1/2}$ through the implementation of Cholesky decomposition.

Given a tensor `X`, Tucker-2 decomposition approximates it as:
$$
\mathbf{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(T)} \times_2 \mathbf{U}^{(S)}
$$
where $\mathcal{G}$ is a core tensor and $\mathbf{U}^{(T)}, \mathbf{U}^{(S)}$ are factor matrices. Then, we present the Tucker2-ALS-Sigma algorithm (See the Algorithm 2 in the paper) which is used to iteratively update these factors to minimize the reconstruction error under Sigma norm.


## Project Structure

```
├── Compute_Covariance.py              # Function to compute Sigma Matrix through Cholesky decomposition of covariance matrix
├── Tucker_Sigma.py            # ALS-Sigma algorithm for Tucker-2
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Install Dependencies
Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run 'Compute_Covariance.py'
This step computes the Sigma matrix from a given model and a dataset.

```bash
python Compute_Covariance.py --model  $path_to_model \
    --transf  $path_to_image_transform_file \
    --root  $path_to_dataset \
    --dataset_files  $dataset_files_path \
    --dataset_labels  $dataset_labels_path \
    --output $output_file_name \
    --workers 1
```

### 3. Run 'Tucker_Sigma.py'
This step runs Tucker2-ALS-Sigma algorithm.
```bash
python Tucker_Sigma.py --tensor_path $path_to_tensor \ 
    --rank $estimated_ranks_of_tensor \
    --sigma_path $path_to_tensor \
    --n_iter_max $number_of_iterations
```


