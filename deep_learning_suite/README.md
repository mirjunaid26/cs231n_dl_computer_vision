# Deep Learning Suite

This directory contains the advanced deep learning components implemented as part of the CS231n Toolkit. It separates these components from the basic linear classifiers found in the project root.

## 📦 Contents

| File | Description |
|------|-------------|
| `layers.py` | Core layers: Affine, ReLU, Softmax, SVM, Loss functions. |
| `optim.py` | Optimization rules: SGD, Momentum, RMSProp, Adam. |
| `batchnorm.py` | Batch Normalization (Train/Test modes, Spatial BN). |
| `dropout.py` | Inverted Dropout. |
| `conv_layers.py` | Naive implementations of Convolution and Max Pooling. |
| `fast_layers.py` | Optimized `im2col` implementations of Conv/Pool. |
| `layer_utils.py` | Convenience layers (e.g., `affine_relu_forward`). |
| `fully_connected_net.py` | Modular Fully Connected Network (arbitrary depth). |
| `cnn_model.py` | 3-Layer Convolutional Neural Network. |
| `trainer.py` | `Solver` class for training models. |
| `pytorch_cifar10.py` | PyTorch implementation of ResNet training on CIFAR-10. |
| `utils.py` | Data loading and gradient checking utilities. |

## 📐 Mathematical Derivations

### 1. Batch Normalization Backward Pass
We implement the alternative backward pass for efficiency.
Let $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$.
The gradient w.r.t input $x$ is:
$$
\frac{\partial L}{\partial x} = \frac{\gamma}{N\sqrt{\sigma^2+\epsilon}} \left( N \frac{\partial L}{\partial \hat{x}} - \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} - \hat{x} \sum_{j=1}^N (\frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j) \right)
$$

### 2. Convolution Gradient (Weight)
For a convolution operation $y = w * x + b$, the gradient w.r.t weights $w$ is the convolution of the input $x$ and the upstream gradient $dout$:
$$ dw = x * dout $$
(Note: handling dimensions correctly involves tensor contraction or `im2col` transpose multiplication).

## 🚀 Usage

### Numpy Models
To train the custom NumPy models, you can run scripts from this directory or import the classes.

**Example (pseudo-code):**
```python
from deep_learning_suite.fully_connected_net import FullyConnectedNet
from deep_learning_suite.trainer import Solver
from deep_learning_suite.utils import get_cifar10_data

data = get_cifar10_data()
model = FullyConnectedNet(hidden_dims=[100, 100], use_batchnorm=True)
solver = Solver(model, data)
solver.train()
```

### PyTorch Model
To run the PyTorch training script:
```bash
python pytorch_cifar10.py
```
This script handles data loading, model definition (Custom ResNet), and training loop using GPU if available.
