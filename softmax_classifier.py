import numpy as np

class SoftmaxClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        (Same as LinearSVM)
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        """
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Softmax loss function, vectorized implementation.
        """
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as svm_loss_vectorized.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train = X.shape[0]

    # 1. Compute scores
    scores = X.dot(W) # (N, C)
    
    # 2. Shift scores for numerical stability
    # Subtract max score for each example
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 3. Compute Softmax
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (N, C)
    
    # 4. Compute Loss
    # -log(correct_class_probs)
    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_logprobs)
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # 5. Compute Gradient
    # dL/ds_k = p_k - 1 (if k=y)
    dscores = probs
    dscores[np.arange(num_train), y] -= 1
    dscores /= num_train
    
    dW = X.T.dot(dscores)
    dW += 2 * reg * W
    
    return loss, dW
