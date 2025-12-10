import numpy as np
from layers import *
from layer_utils import *
from batchnorm import *
from dropout import *

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    
    Learnable parameters are stored in the self.params dictionary.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize parameters
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)] = np.random.randn(dims[i], dims[i+1]) * weight_scale
            self.params['b%d' % (i + 1)] = np.zeros(dims[i+1])
            
            if self.use_batchnorm and i < self.num_layers - 1:
                self.params['gamma%d' % (i + 1)] = np.ones(dims[i+1])
                self.params['beta%d' % (i + 1)] = np.zeros(dims[i+1])

        # Dropout param
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # Batchnorm params
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all params to dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm / dropout
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        
        # Forward Pass
        layers_cache = {}
        out = X
        
        for i in range(self.num_layers - 1): # Hidden layers
            # Affine
            w, b = self.params['W%d'%(i+1)], self.params['b%d'%(i+1)]
            out, affine_cache = affine_forward(out, w, b)
            
            # Batchnorm
            bn_cache = None
            if self.use_batchnorm:
                out, bn_cache = batchnorm_forward(out, self.params['gamma%d'%(i+1)], 
                                                  self.params['beta%d'%(i+1)], 
                                                  self.bn_params[i])
            
            # ReLU
            out, relu_cache = relu_forward(out)
            
            # Dropout
            drop_cache = None
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
                
            layers_cache[i] = (affine_cache, bn_cache, relu_cache, drop_cache)
            
        # Final Output Layer (Affine only)
        w, b = self.params['W%d'%self.num_layers], self.params['b%d'%self.num_layers]
        scores, last_affine_cache = affine_forward(out, w, b)

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        
        # Loss
        loss, dscores = softmax_loss(scores, y)
        
        # Regularization
        for i in range(1, self.num_layers + 1):
             loss += 0.5 * self.reg * np.sum(self.params['W%d' % i]**2)

        # Backward Pass
        dout = dscores
        
        # Last layer backward
        dx, dw, db = affine_backward(dout, last_affine_cache)
        grads['W%d'%self.num_layers] = dw + self.reg * self.params['W%d'%self.num_layers]
        grads['b%d'%self.num_layers] = db
        dout = dx
        
        # Hidden layers backward loop (reverse)
        for i in range(self.num_layers - 2, -1, -1):
            affine_cache, bn_cache, relu_cache, drop_cache = layers_cache[i]
            
            # Dropout
            if self.use_dropout:
                dout = dropout_backward(dout, drop_cache)
            
            # ReLU
            dout = relu_backward(dout, relu_cache)
            
            # Batchnorm
            if self.use_batchnorm:
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
                grads['gamma%d'%(i+1)] = dgamma
                grads['beta%d'%(i+1)] = dbeta
            
            # Affine
            dx, dw, db = affine_backward(dout, affine_cache)
            grads['W%d'%(i+1)] = dw + self.reg * self.params['W%d'%(i+1)]
            grads['b%d'%(i+1)] = db
            dout = dx
            
        return loss, grads
