import numpy as np

class kMeansFromScratch:
    # Step 1: Initialize max_cluster, max_iter, tolerance, centroids, labels
    def __init__(self, n_cluster=3, max_iter=100, tolerance=1e-4, random_state=42):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

        self.centroids = None
        self.labels = None

    # Step 2: Intialize centroids to random choice
    def initialize_centroids(self, X):
        indices = np.random.choice(len(X), self.n_cluster, replace=False)
        return X[indices]

    # Step 3: Compute the distances
    def compute_distances(self, X):
        distances = np.zeros((len(X), self.n_cluster))

        for i, x in enumerate(X):
            for k, centroid in enumerate(self.centroids):
                distances[i,k] = np.sqrt(np.sum((x - centroid)**2))

        return distances

    
    # Step 4: Assign cluster by checking the argmin in the axis=1 for each row
    def assign_cluster(self, X):
        dist = self.compute_distances(X)
        return np.argmin(dist, axis =1)

    # Step 5: update centroid value, by calculating the mean for each unique values.
    def update_centroid(self, X, labels):
        new_centroids = np.zeros_like(self.n_cluster)

        for k in range(self.n_cluster):
            cluster_points = X[labels==k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(
                    cluster_points, axis = 0
                )
            else:
                new_centroids[k] = self.centroids[k]

        return new_centroids

    # Check for convergence
    def has_converged(self, old_centroids, new_centroids):
        centroid_shift = np.linalg.norm(new_centroids - old_centroids)
        return centroid_shift < self.tolerance

    # Write the fit function
    # intialize centroids --> iterate --> centroid_labels --> new_centroid --> convergence --> repeat
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()

            # Assign label
            self.labels = self.assign_cluster(X)

            # Update step
            self.centroids = self.update_centroid(X, self.labels)

            # Convergence
            if self.has_converged(old_centroids, self.centroids):
                print("Centroids converged")
                break

        return self

    def predict(self, X):
        return self.assign_cluster(X)


    def inertia(self, X):
        dist = self.compute_distances(X)
        min_distances = np.min(dist, axis =1)
        return np.sum(min_distances)
        
    
