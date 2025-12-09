import numpy as np

def rgb2gray(rgb):
    """Convert RGB image to grayscale."""
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color history for an image using Hue.
    To be simple and NumPy-only, we might just use RGB histograms if HSV conversion 
    is too complex to implement cheaply.
    Standard CS231n features.py uses matplotlib.colors.rgb_to_hsv usually.
    Let's stick to RGB histograms to avoid external dependencies or complex color math 
    if strictly NumPy.
    But prompt asks for "Color histogram features".
    Layered RGB histogram is safer.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin+1)
    
    # im is (H, W, 3)
    # We want a feature vector.
    # Histogram for R, G, B independently might be useful.
    
    im_flat = im.reshape(-1, 3)
    
    hist_r, _ = np.histogram(im_flat[:, 0], bins=bins)
    hist_g, _ = np.histogram(im_flat[:, 1], bins=bins)
    hist_b, _ = np.histogram(im_flat[:, 2], bins=bins)
    
    hist = np.concatenate([hist_r, hist_g, hist_b])
    
    if normalized:
        hist = hist.astype(float) / np.sum(hist)
        
    return hist

def hog_feature(im):
    """
    Compute HOG feature for an image. 
    Simplified implementation using NumPy.
    
    1. Grayscale
    2. Gradients
    3. 8x8 cells, 9 bins.
    """
    # 1. Convert to float/gray
    if im.ndim == 3:
        im = rgb2gray(im)
    
    im = im.astype(float)
    
    # 2. Gradients (simple differencing)
    gx = np.empty(im.shape, dtype=np.double)
    gx[:, 0] = 0
    gx[:, -1] = 0
    gx[:, 1:-1] = im[:, 2:] - im[:, :-2]
    
    gy = np.empty(im.shape, dtype=np.double)
    gy[0, :] = 0
    gy[-1, :] = 0
    gy[1:-1, :] = im[2:, :] - im[:-2, :]
    
    mag = np.sqrt(gx**2 + gy**2)
    ori = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # 3. Spatial binning (Cells)
    # Assume 32x32 image -> 4x4 cells of 8x8 pixels
    cell_size = 8
    n_cells_x = im.shape[1] // cell_size
    n_cells_y = im.shape[0] // cell_size
    n_bins = 9
    
    hog = np.zeros((n_cells_y, n_cells_x, n_bins))
    
    bin_width = 180 / n_bins
    
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ori = ori[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            # Weighted histogram
            # Simple voting logic:
            # For each pixel, add mag to the bin corresponding to ori
            for p_y in range(cell_size):
                for p_x in range(cell_size):
                    angle = cell_ori[p_y, p_x]
                    magnitude = cell_mag[p_y, p_x]
                    
                    bin_idx = int(angle / bin_width) % n_bins
                    hog[i, j, bin_idx] += magnitude
                    
    # Flatten
    return hog.ravel()

def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions, can compute the features
    for all images.

    Inputs:
    - imgs: N x H x W x C array of pixel data.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    - features: N x F array of features, where F is the sum of F_i.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])
    
    feature_dims = []
    
    # Process first image to determine feature dims
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        first_image_features.append(feats)
        feature_dims.append(feats.size)
    
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.concatenate(first_image_features)
    
    for i in range(1, num_images):
        idx = 0
        current_features = []
        for feature_fn in feature_fns:
             # imgs[i] might be (1, 32, 32, 3) or (32, 32, 3)
             # Squeeze helps ensure (H, W, C)
            fts = feature_fn(imgs[i].squeeze())
            current_features.append(fts)
            
        imgs_features[i] = np.concatenate(current_features)
        
        if verbose and i % 1000 == 0:
            print('Done extracting features for %d / %d images' % (i, num_images))
            
    return imgs_features
