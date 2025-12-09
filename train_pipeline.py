import numpy as np
import data_utils
from knn_classifier import KNearestNeighbor
from svm_classifier import LinearSVM
from softmax_classifier import SoftmaxClassifier
from two_layer_net import TwoLayerNet
from features import extract_features, hog_feature, color_histogram_hsv
from evaluate import evaluate_classifier, print_comparison_table

def run_pipeline():
    print("Loading data...")
    # Load data
    data = data_utils.get_cifar10_data(cifar10_dir='cifar-10-batches-py', num_training=49000, num_validation=500, num_test=500)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Val data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Flatten for linear classifiers
    X_train_flat = data_utils.flatten_images(X_train)
    X_val_flat = data_utils.flatten_images(X_val)
    X_test_flat = data_utils.flatten_images(X_test)
    
    results = []
    
    # ================================================================
    # 1. Raw Pixels
    # ================================================================
    print("\n--- Training on Raw Pixels ---")
    
    # kNN
    # Uses a subset because it's slow
    print("Training kNN (k=5)...")
    knn = KNearestNeighbor()
    knn.train(X_train_flat[:5000], y_train[:5000]) # smaller train set for kNN speed
    # We predict on full val set
    acc = evaluate_classifier(knn, X_val_flat, y_val, batch_size=100)
    print(f"kNN Validation Accuracy: {acc}")
    results.append({'Model': 'kNN', 'Feature': 'Raw Pixels', 'Train Acc': 1.0, 'Val Acc': acc})
    
    # SVM
    print("Training SVM...")
    svm = LinearSVM()
    loss_hist = svm.train(X_train_flat, y_train, learning_rate=1e-7, reg=2.5e4, num_iters=1500, verbose=False)
    train_acc = evaluate_classifier(svm, X_train_flat, y_train)
    val_acc = evaluate_classifier(svm, X_val_flat, y_val)
    print(f"SVM Validation Accuracy: {val_acc}")
    results.append({'Model': 'SVM', 'Feature': 'Raw Pixels', 'Train Acc': train_acc, 'Val Acc': val_acc})
    
    # Softmax
    print("Training Softmax...")
    softmax = SoftmaxClassifier()
    loss_hist = softmax.train(X_train_flat, y_train, learning_rate=1e-7, reg=2.5e4, num_iters=1500, verbose=False)
    train_acc = evaluate_classifier(softmax, X_train_flat, y_train)
    val_acc = evaluate_classifier(softmax, X_val_flat, y_val)
    print(f"Softmax Validation Accuracy: {val_acc}")
    results.append({'Model': 'Softmax', 'Feature': 'Raw Pixels', 'Train Acc': train_acc, 'Val Acc': val_acc})
    
    # Two Layer Net
    print("Training Two-Layer Net...")
    input_dim = X_train_flat.shape[1]
    hidden_dim = 50
    num_classes = 10
    net = TwoLayerNet(input_dim, hidden_dim, num_classes)
    stats = net.train(X_train_flat, y_train, X_val_flat, y_val,
                      num_iters=1000, batch_size=200,
                      learning_rate=1e-4, learning_rate_decay=0.95,
                      reg=0.25, verbose=False)
    
    print(f"Net Validation Accuracy: {stats['val_acc_history'][-1]}")
    results.append({'Model': 'TwoLayerNet', 'Feature': 'Raw Pixels', 'Train Acc': stats['train_acc_history'][-1], 'Val Acc': stats['val_acc_history'][-1]})
    
    
    # ================================================================
    # 2. Features (HOG + Color Hist)
    # ================================================================
    print("\n--- Extracting Features ---")
    
    # Define feature functions
    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=10)]
    
    X_train_feats = extract_features(X_train, feature_fns, verbose=True)
    X_val_feats = extract_features(X_val, feature_fns)
    X_test_feats = extract_features(X_test, feature_fns)
    
    # Normalization implies zero mean and unit variance usually? Or typically we just mean centering
    # Mean centering
    mean_feat = np.mean(X_train_feats, axis=0)
    X_train_feats -= mean_feat
    X_val_feats -= mean_feat
    X_test_feats -= mean_feat
    
    # Prepare simpler variance scaling
    # std_feat = np.std(X_train_feats, axis=0)
    # X_train_feats /= std_feat
    # X_val_feats /= std_feat
    # X_test_feats /= std_feat
    
    # Add bias dimension for linear classifiers
    X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
    X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
    X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])
    
    print(f"Feature shape: {X_train_feats.shape}")
    
    # SVM on Features
    print("Training SVM on Features...")
    svm_feats = LinearSVM()
    # Tuned hyperparameters for features might differ
    svm_feats.train(X_train_feats, y_train, learning_rate=1e-8, reg=5e4, num_iters=1500, verbose=False)
    train_acc = evaluate_classifier(svm_feats, X_train_feats, y_train)
    val_acc = evaluate_classifier(svm_feats, X_val_feats, y_val)
    print(f"SVM (Feats) Validation Accuracy: {val_acc}")
    results.append({'Model': 'SVM', 'Feature': 'HOG+Color', 'Train Acc': train_acc, 'Val Acc': val_acc})
    
    # Softmax on Features
    print("Training Softmax on Features...")
    sf_feats = SoftmaxClassifier()
    sf_feats.train(X_train_feats, y_train, learning_rate=1e-8, reg=5e4, num_iters=1500, verbose=False)
    train_acc = evaluate_classifier(sf_feats, X_train_feats, y_train)
    val_acc = evaluate_classifier(sf_feats, X_val_feats, y_val)
    print(f"Softmax (Feats) Validation Accuracy: {val_acc}")
    results.append({'Model': 'Softmax', 'Feature': 'HOG+Color', 'Train Acc': train_acc, 'Val Acc': val_acc})

    # Neural Net on Features
    print("Training Two-Layer Net on Features...")
    input_dim = X_train_feats.shape[1]
    net_feats = TwoLayerNet(input_dim, hidden_dim, num_classes)
    stats = net_feats.train(X_train_feats, y_train, X_val_feats, y_val,
                      num_iters=1000, batch_size=200,
                      learning_rate=1.0, learning_rate_decay=0.95, # Higher LR for features usually
                      reg=0.01, verbose=False)
    print(f"Net (Feats) Validation Accuracy: {stats['val_acc_history'][-1]}")
    results.append({'Model': 'TwoLayerNet', 'Feature': 'HOG+Color', 'Train Acc': stats['train_acc_history'][-1], 'Val Acc': stats['val_acc_history'][-1]})

    print_comparison_table(results)

if __name__ == "__main__":
    run_pipeline()
