# CS231n Computer Vision Project

This repository contains implementations for the CS231n Deep Learning for Computer Vision course. It is organized into two main sections:

1.  **Linear Classifiers**: The foundational implementations located in the root directory.
2.  **Deep Learning Suite**: Advanced neural network components located in `deep_learning_suite/`.

## 📂 Project Structure

```
cs231n_dl_computer_vision/
│
├── Deep Learning Suite
│   └── deep_learning_suite/   # <--- NEW: Advanced Models & Layers
│       ├── layers.py, optim.py, batchnorm.py, ...
│       ├── fully_connected_net.py
│       ├── cnn_model.py
│       ├── pytorch_cifar10.py
│       └── README.md          # Documentation for the suite
│
├── Linear Classifiers (Root)
│   ├── knn_classifier.py
│   ├── svm_classifier.py
│   ├── softmax_classifier.py
│   ├── two_layer_net.py
│   └── train_pipeline.py
│
└── environment.yml            # Environment setup
```

## 🚀 Setup

1.  **Install Requirements**
    Ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate cs231n
    ```

2.  **Download Dataset**
    The scripts expect `cifar-10-batches-py` in the root directory.

## 🏃 Usage

### 1. Linear Classifiers
To run the original training pipeline for kNN, SVM, Softmax, and simple Neural Net:
```bash
python train_pipeline.py
```

### 2. Deep Learning Suite
To use the advanced components (BatchNorm, Dropout, CNNs, PyTorch), please navigate to the `deep_learning_suite/` directory or refer to its [README](deep_learning_suite/README.md).

For PyTorch training:
```bash
python deep_learning_suite/pytorch_cifar10.py
```
