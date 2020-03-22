import numpy as np 

# ==================================KMeans==============================================
class KMeans(object):
    '''
    An simplified implementation of KMeans based on scikit-learn for algorithm understanding.
    This code is only used for the author(houwenxin)'s own study. No commercial use.

    Parameters:
    -----------
    n_clusters : int, #Clusters (K)
    max_iter : int, Max number of iterations, default=300
    tol : float, tolerance of total center shift (total change of centers after each iteration), default=1e-4.
    '''
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters # Number of clusters
        self.max_iter = max_iter # Maximum number of iteration.
        self.tol = tol

    def fit(self, X, sample_weights=None):
        '''
        Fit the input data.

        Parameters:
        ------------
        X : np.array, shape (n_samples, n_features)
        sample_weights : np.array, shape (n_samples,)


        Returns:
        ------------
        best_labels : np.array, shape (n_samples,)
        best_inertia: float, best sum of distances of samples to their closest centers.
        best_centers : np.array, shape (n_clusters, n_features)
        best_n_iter : int, #Iteration to reach the best result.
        '''
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0], dtype=X.dtype)

        best_inertia, best_labels, best_centers = None, None, None  

        centers = self._init_centroids(X, self.n_clusters)
        distances = np.zeros((X.shape[0],), dtype=X.dtype)

        for i in range(self.max_iter):
            old_centers = centers.copy()
            labels, inertia = self._labels_inertia(X, centers, distances)

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()

            # Re-assign centers.
            centers = self._centers(X, labels, self.n_clusters, sample_weights)

            center_shift_total = ((centers - old_centers) ** 2).sum()
            if center_shift_total <= self.tol:
                print(f"Converge at iteration={i}. Center shift={center_shift_total} within tolerance={self.tol}.")
                break
        
        if center_shift_total > 0:
            best_labels, best_inertia = self._labels_inertia(X, best_centers, distances)
        best_n_iter = i + 1

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def _init_centroids(self, X, n_clusters):
        '''
        Initialize centroids.

        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
        n_clusters : int, #Clusters == #Centers 


        Returns:
        -----------
        centers : np.array, shape (n_clusters, n_features)
        '''
        n_samples = X.shape[0]
        if n_samples < n_clusters:
            raise ValueError(f'#Samples={n_samples} should be larger than K={n_clusters} (#Centers)')
        # Randomly assign centers
        indices = np.random.permutation(n_samples)[:n_clusters]
        centers = X[indices]
        return centers

    def _labels_inertia(self, X, centers, distances):
        '''
        Assign labels and return the sum of distances of samples to their closest centers.

        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
        centers: np.array, shape (n_centers == n_clusters, n_features)
        distances: np.array, shape (n_samples, ) each samples's distance to their closest centers.


        Returns:
        -----------
        labels : np.array, shape (n_samples, )
        inertia : float, total eucludiean distances of samples to their closest centers.
        '''
        n_samples = X.shape[0]
        n_centers = centers.shape[0]
        inertia = 0.0
        labels = np.full(n_samples, -1, np.int32)
        for sample_idx in range(n_samples):
            min_dist = -1 
            for center_idx in range(n_centers):
                dist = ((X[sample_idx] - centers[center_idx]) ** 2).sum()
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    labels[sample_idx] = center_idx
            distances[sample_idx] = min_dist
            inertia += min_dist
        return labels, inertia

    def _centers(self, X, labels, n_clusters, sample_weights):
        '''
        Calculate the centers given samples and corresponding labels.

        Parameters:
        ------------
        X : np.array, shape (n_samples, n_features)
        labels : np.array, shape (n_samples,)
        n_clusters : int, #Clusters
        sample_weights : np.array, shape (n_samples,)

        Returns:
        ------------
        centers : np.array, shape (n_clusters, n_features)
        '''
        n_samples = X.shape[0]
        n_features = X.shape[1]

        dtype = np.float32
        centers = np.zeros((n_clusters, n_features), dtype=dtype)
        weight_in_clusters = np.zeros((n_clusters,), dtype=dtype)

        for sample_idx in range(n_samples):
            c = labels[sample_idx]
            weight_in_clusters[c] += sample_weights[sample_idx]

        for sample_idx in range(n_samples):
            # for feature_idx in range(n_features):
            #     centers[labels[sample_idx], feature_idx] += X[sample_idx, feature_idx]
            centers[labels[sample_idx], :] += X[sample_idx, :] # easier to understand.

        centers /= weight_in_clusters[:, np.newaxis]
        return centers

# =====================================Simple Test=======================================================

def test():
    X = np.array(
        [[1, 2], [1, 4], [1, 0],
        [10, 2], [10, 4], [10, 0]]
    )
    kmeans = KMeans(n_clusters=2, max_iter=100, tol=1e-5).fit(X)
    print(kmeans.labels_, '\n', kmeans.cluster_centers_)
    
if __name__ == "__main__":
    test()
