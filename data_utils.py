import numpy as np
import os
import pickle

def load_cifar_batch(filename):
    """
    Loads a single batch of CIFAR-10 data.
    
    Args:
        filename: Path to the batch file.
        
    Returns:
        X: (N, 3072) float array, in [0, 1] range (actually usually u8, will convert)
        Y: (N,) int array of labels
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_cifar10(root):
    """
    Loads the full CIFAR-10 dataset.
    
    Args:
        root: Directory containing the CIFAR-10 data files (data_batch_1, etc.)
        
    Returns:
        X_train: (50000, 32, 32, 3)
        y_train: (50000,)
        X_test:  (10000, 32, 32, 3)
        y_test:  (10000,)
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b, ))
        if not os.path.exists(f):
            print(f"File {f} not found. Returning None.")
            return None, None, None, None
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    
    X_test, y_test = load_cifar_batch(os.path.join(root, 'test_batch'))
    
    return X_train, y_train, X_test, y_test

def get_cifar10_data(cifar10_dir='cifar-10-batches-py', 
                     num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. 
    
    If dataset not found, generates synthetic data.
    """
    # Load the raw CIFAR-10 data
    try:
        X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)
    except Exception:
        X_train = None

    if X_train is None:
        print(f"Could not find CIFAR-10 at {cifar10_dir}. Generating synthetic data.")
        # Synthetic data: 32x32x3 images
        X_train = np.random.randn(50000, 32, 32, 3) * 10
        y_train = np.random.randint(0, 10, 50000)
        X_test = np.random.randn(10000, 32, 32, 3) * 10
        y_test = np.random.randint(0, 10, 10000)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
    
    # Transpose so that channels come first (N, C, H, W) for some implementations?
    # Or keep (N, H, W, C)?
    # Standard CS231n assignments usually flatten to (N, D).
    # Some also use (N, C, H, W) for conv nets.
    # For this assignment, we mostly need (N, D) for linear classifiers.
    # But for TwoLayerNet we might want flexibility.
    # I will return (N, 32, 32, 3) and provide a flatten utility.
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'mean_image': mean_image if subtract_mean else None
    }

def flatten_images(X):
    """
    Flatten images from (N, H, W, C) to (N, D).
    """
    return X.reshape(X.shape[0], -1)

class BatchInfo:
    """ Iterator for batch sampling """
    def __init__(self, X, y, batch_size=256, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_examples = X.shape[0]
        self.idx = 0
        self.indices = np.arange(self.num_examples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_examples:
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration
            
        batch_indices = self.indices[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        return X_batch, y_batch

def random_batch(X, y, batch_size=256):
    """
    Simply returns a random batch for SGD.
    """
    num_train = X.shape[0]
    batch_mask = np.random.choice(num_train, batch_size)
    X_batch = X[batch_mask]
    y_batch = y[batch_mask]
    return X_batch, y_batch
