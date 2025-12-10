import numpy as np
from layers import *
from fast_layers import *
from layer_utils import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W).
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        
        # Conv layer params
        # W1: (F, C, HH, WW)
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        
        # Output of Conv/Pool size
        # Assuming stride 1 pad (F-1)/2 for conv -> preserves size
        # Pool 2x2 stride 2 -> divides size by 2
        pad = (filter_size - 1) // 2
        H_out = H // 2
        W_out = W // 2
        # F * H/2 * W/2
        flattened_dim = num_filters * H_out * W_out
        
        # Hidden affine layer params
        self.params['W2'] = np.random.randn(flattened_dim, hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        
        # Output affine layer params
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluates loss and gradient for the three-layer convolutional network.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # Pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        
        # Forward Pass
        # 1. Conv - ReLU - Pool
        out_pool, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        
        # 2. Affine - ReLU
        out_affine1, cache2 = affine_relu_forward(out_pool, W2, b2)
        
        # 3. Affine
        scores, cache3 = affine_forward(out_affine1, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}
        
        # Loss
        loss, dscores = softmax_loss(scores, y)
        
        # L2 Regularization
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        
        # Backward Pass
        # 3. Affine
        dx3, dw3, db3 = affine_backward(dscores, cache3)
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3
        
        # 2. Affine - ReLU
        dx2, dw2, db2 = affine_relu_backward(dx3, cache2)
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2
        
        # 1. Conv - ReLU - Pool
        dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache1)
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1
        
        return loss, grads
