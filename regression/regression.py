import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.regular_stregth = 0.01

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for epoch in range(self.epochs):
            predictions = X@self.weights + self.bias
            error = predictions - y
            dw = 2 * np.mean(X.T @ error)
            db = 2 * np.mean(np.sum(error))

            self.weights -= self.lr * dw
            self.bias -= self.lr*db

    def fit_regularize(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):
            pred = X @ self.weights + self.bias
            error = pred - y

            dw = (2/n_samples) * (X.T @ error)
            dw += (self.regular_stregth/n_samples) * self.weights
            db = (2/n_samples)*(np.sum(error))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return X @ self.weights + self.bias

    def mse(self, y_true, y_pred):
        return float(np.mean((y_true - y_pred)**2))

    def rmse(self, y_true, y_pred):
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def r2(self, y_true, y_pred):
        ss_res = np.sum((y_true-y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_pred))**2)

        return float(1-(ss_res/ss_tot))


    

