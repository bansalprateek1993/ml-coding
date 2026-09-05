import numpy as np

class lrMutiClass:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.reg_stregth = 0.01

    def softmax(self, z):
        exp_z = np.exp(z)
        den = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z/den

    def softmax_opt(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        den = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z/den

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))        
        self.bias = np.zeros(n_classes)

        Y = np.eye(n_classes)[y]

        for epoch in len(self.epochs):
            logits = X @ self.weights + self.bias
            y_pred = self.softmax(logits)
            error = y_pred - y
            dw = np.mean(X.T @ error)

            # With L2 regularization
            dw += (self.reg_stregth/n_samples)*self.weights
            db = np.mean(np.sum(error, axis=0))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_probs(self, X):
        logits = X@self.weights + self.bias
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_probs(X)
        return np.argmax(probs, axis=1)
    
        

