import numpy as np

def compute_accuracy(y_true, y_pred):
    """
    Computes accuracy of predictions.
    """
    return np.mean(y_true == y_pred)

def evaluate_classifier(classifier, X, y, batch_size=None):
    """
    Runs prediction and computes accuracy.
    If batch_size is provided, predicts in batches to save memory (useful for large kNN).
    """
    if batch_size is None:
        y_pred = classifier.predict(X)
    else:
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        for i in range(0, num_test, batch_size):
            end = min(i + batch_size, num_test)
            y_pred[i:end] = classifier.predict(X[i:end])
            
    return compute_accuracy(y, y_pred)

def print_comparison_table(results):
    """
    Prints a comparison table of accuracies.
    
    Args:
        results: List of dictionaries with keys:
                 'Model', 'Feature', 'Train Acc', 'Val Acc', 'Test Acc' (optional)
    """
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'Feature':<15} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10}")
    print("-" * 80)
    
    for res in results:
        test_acc = res.get('Test Acc', '-')
        if isinstance(test_acc, float):
             test_acc = f"{test_acc:.3f}"
        
        print(f"{res['Model']:<20} | {res['Feature']:<15} | "
              f"{res['Train Acc']:.3f}      | {res['Val Acc']:.3f}      | {test_acc:<10}")
    print("="*80 + "\n")
