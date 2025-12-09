import numpy as np

class LinearSVM(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Returns:
        - loss_history: A list containing the value of the loss function at each
          training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # Lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Sample batch_size elements from the training data and their labels
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data (Wait, standard is N x D)
          Let's assume N x D to match train (X.shape[0] samples).

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Structured SVM loss function, vectorized implementation.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of data.
        - y_batch: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

        Returns a tuple of:
        - loss: A float, the total loss (data loss + regularization loss)
        - dW: A numpy array of shape (D, C) containing the gradient with respect to W.
        """
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss: (float)
    - dW: A numpy array of shape (D, C) containing the gradient of loss with respect to W.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_train = X.shape[0]
    
    # 1. Compute scores
    scores = X.dot(W) # (N, C)
    
    # 2. Compute margins
    # Select the scores of the correct classes
    # y is (N,)
    correct_class_scores = scores[np.arange(num_train), y] # (N,)
    correct_class_scores = correct_class_scores.reshape(num_train, 1) # (N, 1)
    
    margins = scores - correct_class_scores + 1.0 # delta = 1
    
    # 3. Apply hinge
    margins[margins < 0] = 0
    
    # 4. Ignore the correct class margins (they should be 0 anyway if logic was slightly diff, 
    # but here (s_y - s_y + 1) = 1, so we must manually zero them)
    margins[np.arange(num_train), y] = 0
    
    # 5. Compute loss
    loss = np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W * W)

    # 6. Compute Gradient
    # For j != y_i: dL/dw_j = x_i * indicator(margin > 0)
    # For j == y_i: dL/dw_y_i = - sum_{j!=y_i} x_i * indicator(margin > 0)
    
    # Binary mask of margins > 0
    binary_mask = margins
    binary_mask[margins > 0] = 1
    
    # Count how many classes contributed to loss for each example
    row_sum = np.sum(binary_mask, axis=1) # (N,)
    
    # For the correct class, subtract the count times X[i]
    binary_mask[np.arange(num_train), y] = -row_sum
    
    # Now simply dot product
    dW = X.T.dot(binary_mask)
    
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
