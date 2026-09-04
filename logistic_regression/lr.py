import numpy as np


class lrFromScratch:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, y):
        return 1/(1+np.exp(-y))

    def bce(self, y, prob):
        eps = 1e-15
        prob = np.clip(prob, eps, 1-eps)
        loss = -np.mean(y*np.log(prob) + (1-y)*np.log(1-prob))
        return float(loss)

    def predict_probs(self, X):
        # print(X.shape)
        # print(self.weights.shape)
        z = X @ self.weights + self.bias
        # print(z.shape)
        return self.sigmoid(z)

    def predict(self, X):
        prob = self.predict_probs(X)
        return (prob >= 0.5).astype(int)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for epoch in range(self.epochs):
            # l1 = X @ self.weights + self.bias
            y_pred = self.predict_probs(X)
            error = y_pred - y

            dw = np.mean(X.T @ error)
            db = np.mean(np.sum(error))

            self.weights -= self.lr * dw
            self.bias -= self.lr*db

    def cls_metrics(self, y_true, y_pred):
        tp = np.sum((y_true==1) & (y_pred==1))
        fp = np.sum((y_true==0) & (y_pred==1))
        tn = np.sum((y_true==0) & (y_pred==0))
        fn = np.sum((y_true==1) & (y_pred==0))

        precision = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        recall = tp/(tp + fn) if (tp+fn) > 0 else 0.0

        f1 = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
        accuracy = (tp + tn)/len(y_true)
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }





