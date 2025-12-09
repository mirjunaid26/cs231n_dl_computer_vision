# Image Classification Pipeline

This package implements a complete image classification pipeline using only Python and NumPy. It includes implementations of k-Nearest Neighbors (kNN), Support Vector Machines (SVM), Softmax Classifiers, and a Two-Layer Neural Network.

## Setup

1.  Ensure you have Python 3 and NumPy installed.
    ```bash
    pip install numpy
    ```

2.  (Optional) Download the CIFAR-10 dataset.
    - Download `cifar-10-python.tar.gz` from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
    - Extract it in the `image_classification` directory.
    - The code expects the folder `cifar-10-batches-py`.
    - **If no dataset is found, the code will automatically generate synthetic data for demonstration.**

## Project Structure

- `data_utils.py`: Utilities for loading CIFAR-10, splitting data, and preprocessing.
- `knn_classifier.py`: k-Nearest Neighbor classifier with vectorized distance computations.
- `svm_classifier.py`: Linear SVM with vectorized hinge loss and SGD training.
- `softmax_classifier.py`: Softmax classifier with vectorized cross-entropy loss and SGD.
- `two_layer_net.py`: Fully connected neural network with ReLU and modular forward/backward passes.
- `features.py`: Feature extraction utilities (Color Histograms, HOG-like features).
- `evaluate.py`: Evaluation metrics and comparison utilities.
- `train_pipeline.py`: Main script to train and evaluate all models.

## Usage

To run the full training pipeline and see comparison results:

```bash
python image_classification/train_pipeline.py
```

## Implementation Details

- **Vectorization**: All loss functions and gradients are fully vectorized using NumPy broadcasting to ensure high performance compared to explicit loops.
- **Modularity**: The Neural Network implementation separates loss computation, parameter updates, and prediction.
- **Features**: The pipeline compares performance on raw pixel data vs. extracted features (HOG and Color Histograms).
