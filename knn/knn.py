import numpy as np

class knnScratch:
    def __init__(self, k_points=5):
        self.k_points = k_points
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        return self

    def predict_one(self, x):
        distance = []

        for i in range(len(self.X_train)):
            dist = np.sum((x - self.X_train[i])**2)
            distance.append((dist, self.y_train[i]))

        distance.sort(key=lambda x: x[0])
        nearest = distance[:self.k_points]
        labels = [label for dist, label in nearest]
        value, counts = np.unique(labels, return_counts = True)
        return value[np.argmax(counts)]

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.predict_one(x))

        return np.array(prediction)

    
