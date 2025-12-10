# Deep Learning Toolkit & CIFAR-10 Challenge

This repository contains a modular, university-level implementation of Deep Learning components from scratch using **Python and NumPy**, culminating in a **PyTorch** solution for the CIFAR-10 dataset.

## 📂 Project Structure

```
cs231n_dl_computer_vision/
│
├── Core (NumPy)
│   ├── layers.py              # Affine, ReLU, Softmax, SVM
│   ├── optim.py               # SGD, Momentum, Adam, RMSProp
│   ├── batchnorm.py           # Batch Normalization (Forward/Backward/Spatial)
│   ├── dropout.py             # Inverted Dropout
│   ├── conv_layers.py         # Naive Convolution & Pooling
│   ├── fast_layers.py         # Optimized Im2Col Convolution
│   ├── layer_utils.py         # Composite layers (e.g., Affine-ReLU)
│   ├── utils.py               # Data loading & Gradient checking
│
├── Models
│   ├── fully_connected_net.py # Arbitrary depth FC Nets
│   ├── cnn_model.py           # 3-Layer CNN
│   ├── trainer.py             # Training loop & Solver
│
├── PyTorch Scale-up
│   └── pytorch_cifar10.py     # Custom ResNet on CIFAR-10 with GPU support
│
├── environment.yml            # Conda environment definition
└── README.md                  # This file
```

## 🚀 Setup

1.  **Create Conda Environment**
    ```bash
    conda env create -f environment.yml
    conda activate cs231n
    ```

2.  **Dataset**
    The scripts will automatically download CIFAR-10. If not, it expects `cifar-10-batches-py` in the `data/` or root directory.

## 🧠 Components & Mathematical Derivations

### 1. Fully Connected Layers

**Forward Pass**: $y = xW + b$

**Backward Pass Gradients**:
Given upstream gradient $\frac{\partial L}{\partial y}$ (shape $N \times M$):
- $\frac{\partial L}{\partial b} = \sum_{i=1}^N \frac{\partial L}{\partial y_i}$
- $\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y}$
- $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$

### 2. Batch Normalization

We implement Batch Normalization to stabilize training.

**Forward**:
$\mu = \frac{1}{N}\sum x_i$
$\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2$
$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
$y_i = \gamma \hat{x}_i + \beta$

**Backward (Alternative Derivation)**:
Instead of backpropagating through every intermediate step, we use the simplified gradient for $\hat{x}$:
$\frac{\partial L}{\partial x} = \frac{1}{N\sqrt{\sigma^2+\epsilon}} \left( N \frac{\partial L}{\partial \hat{x}} - \sum \frac{\partial L}{\partial \hat{x}} - \hat{x} \sum (\frac{\partial L}{\partial \hat{x}} \cdot \hat{x}) \right)$
Multiply by $\gamma$ to get final gradient w.r.t input.

### 3. Dropout (Inverted)

**Train**: $mask \sim Bernoulli(p)$, $y = x \odot mask / p$
**Test**: $y = x$
The division by $p$ during training ensures the expected output magnitude remains constant, so no scaling is needed at test time.

### 4. Convolution (Im2Col Optimization)

Naive convolution with nested loops is slow ($O(N \cdot C \cdot H \cdot W \cdot F \cdot HH \cdot WW)$).
We optimize this using `im2col`:
1.  **Im2Col**: Reshape input image patches into columns of a large matrix $X_{col}$.
2.  **Filter**: Reshape filters into rows of a matrix $W_{row}$.
3.  **GEMM**: Compute $Out = W_{row} \cdot X_{col}$.
4.  **Col2Im**: Reshape result back to $(N, F, H', W')$.

## 🏃 Usage

### Train NumPy Models
You can script the training using the `Solver` class in `trainer.py` or import models in your scripts.

```python
from fully_connected_net import FullyConnectedNet
from trainer import Solver
from utils import get_cifar10_data

data = get_cifar10_data()
model = FullyConnectedNet(hidden_dims=[100, 100], dropout=0.5, use_batchnorm=True)
solver = Solver(model, data, update_rule='adam', num_epochs=10)
solver.train()
```

### Train PyTorch Model
We provide a high-performance training script with a Custom ResNet architecture.

```bash
python pytorch_cifar10.py
```
**Features**:
- Data Augmentation (RandomCrop, Flip)
- Adam Optimizer with MultiStepLR
- Checkpointing best models
- Validation Accuracy logging
