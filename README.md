# Image Classification Pipeline

This package implements a complete image classification pipeline using only Python and NumPy. It includes implementations of k-Nearest Neighbors (kNN), Support Vector Machines (SVM), Softmax Classifiers, and a Two-Layer Neural Network.

## Setup

1.  **Install Requirements**
    Ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate cs231n
    ```

2.  **(Optional) Download the CIFAR-10 dataset**
    - Download `cifar-10-python.tar.gz` from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
    - Extract it in the project root directory.
    - The code expects the folder `cifar-10-batches-py`.
    - **Note:** If no dataset is found, the code will automatically generate synthetic data for demonstration purposes.

## Project Structure

The project is organized as follows:

- **Classifiers**:
    - `knn_classifier.py`: k-Nearest Neighbor classifier with vectorized distance computations.
    - `svm_classifier.py`: Linear SVM with vectorized hinge loss and SGD training.
    - `softmax_classifier.py`: Softmax classifier with vectorized cross-entropy loss and SGD.
    - `two_layer_net.py`: Fully connected neural network with ReLU and modular forward/backward passes.
- **Utilities**:
    - `data_utils.py`: Utilities for loading CIFAR-10, splitting data, and preprocessing.
    - `features.py`: Feature extraction utilities (Color Histograms, HOG-like features).
    - `evaluate.py`: Evaluation metrics and comparison utilities.
- **Main**:
    - `train_pipeline.py`: Main script to train and evaluate all models.

## Usage

To run the full training pipeline and see comparison results:

```bash
python train_pipeline.py
```

### Expected Output

The script will:
1.  Load the data (or generate synthetic data).
2.  Train each model (kNN, SVM, Softmax, Two-Layer Net) on **Raw Pixels**.
3.  Train linear classifiers and the neural net on **Extracted Features** (HOG + Color Histograms).
4.  Print a comparison table of training and validation accuracies.

## Implementation Details

- **Vectorization**: All loss functions and gradients are fully vectorized using NumPy broadcasting to ensure high performance compared to explicit loops.
- **Modularity**: The Neural Network implementation separates loss computation, parameter updates, and prediction.
- **Features**: The pipeline compares performance on raw pixel data vs. extracted features (HOG and Color Histograms).
