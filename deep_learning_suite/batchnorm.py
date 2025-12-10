import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also split an exponentially decaying running mean of the
    mean and variance of the input. These statistics are collected but not
    used during training. During testing we use the running mean and variance
    to normalize.

    The normalization strategy is naive integration:
    x_hat = (x - mean) / sqrt(var + eps)
    out = gamma * x_hat + beta

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var: Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        # Step 1: Compute sample mean
        mu = np.mean(x, axis=0)
        
        # Step 2: Compute sample variance
        var = np.var(x, axis=0) # biased variance is standard for BN
        
        # Step 3: Normalize
        x_hat = (x - mu) / np.sqrt(var + eps)
        
        # Step 4: Scale and shift
        out = gamma * x_hat + beta
        
        # Update running stats
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        
        cache = (x, x_hat, mu, var, gamma, eps) # Store for backward pass
        
    elif mode == 'test':
        # Use running stats
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, x_hat, mu, var, gamma, eps = cache
    N, D = dout.shape
    
    # Gradient w.r.t beta and gamma is easy
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    # Derivative w.r.t x_hat
    dx_hat = dout * gamma
    
    # Derivative w.r.t variance
    # dL/dvar = sum(dL/dx_hat * dx_hat/dvar)
    # dx_hat/dvar = (x - mu) * (-0.5) * (var + eps)^(-1.5)
    ivar = 1.0 / np.sqrt(var + eps)
    dxmu1 = x - mu
    dvar = np.sum(dx_hat * dxmu1 * -0.5 * (ivar**3), axis=0)
    
    # Derivative w.r.t mean
    # dL/dmu = sum(dL/dx_hat * dx_hat/dmu) + dL/dvar * dvar/dmu
    # dx_hat/dmu = -1 / sqrt(var + eps)
    # dvar/dmu = sum(-2 * (x - mu)) / N
    dmu = np.sum(dx_hat * -ivar, axis=0) + dvar * np.sum(-2 * dxmu1, axis=0) / N
    
    # Derivative w.r.t x
    # dL/dx = dL/dx_hat * dx_hat/dx + dL/dvar * dvar/dx + dL/dmu * dmu/dx
    # dx_hat/dx = 1 / sqrt(var + eps)
    # dvar/dx = 2 * (x - mu) / N
    # dmu/dx = 1 / N
    dx = dx_hat * ivar + dvar * 2 * dxmu1 / N + dmu / N
    
    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    Uses the simplified derivation:
    dx = (1/N) * gamma * (var + eps)^(-1/2) * [ N*dout - sum(dout) - x_hat*sum(dout*x_hat) ]
    """
    x, x_hat, mu, var, gamma, eps = cache
    N, D = dout.shape
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    ivar = 1.0 / np.sqrt(var + eps)
    
    dx = (1.0 / N) * gamma * ivar * (
        N * dout - np.sum(dout, axis=0) - x_hat * np.sum(dout * x_hat, axis=0)
    )
    
    return dx, dgamma, dbeta

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      
    Returns a tuple of:
    - out: Output data of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    
    # Reshape to (N*H*W, C) to reuse standard BN
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # Reshape back to (N, C, H, W)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    
    # Reshape to (N*H*W, C)
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    
    dx_reshaped, dgamma, dbeta = batchnorm_backward_alt(dout_reshaped, cache)
    
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    return dx, dgamma, dbeta
