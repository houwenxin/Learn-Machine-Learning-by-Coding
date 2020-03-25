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
    def __init__(self, n_clusters, init='random', max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters # Number of clusters
        self.max_iter = max_iter # Maximum number of iteration.
        self.tol = tol
        self.init = init

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

        centers = self._init_centroids(X, self.n_clusters, self.init)
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

    def _init_centroids(self, X, n_clusters, init):
        '''
        Initialize centroids.

        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
        n_clusters : int, #Clusters == #Centers 
        init : str, 'random' or 'kmeans++'


        Returns:
        -----------
        centers : np.array, shape (n_clusters, n_features)
        '''
        if init == 'random':
            n_samples = X.shape[0]
            if n_samples < n_clusters:
                raise ValueError(f'#Samples={n_samples} should be larger than K={n_clusters} (#Centers)')
            # Randomly assign centers
            indices = np.random.permutation(n_samples)[:n_clusters]
            centers = X[indices]

        elif init == 'kmeans++':
            centers = self._k_init(X, n_clusters)
        return centers

    def _k_init(self, X, n_clusters, n_local_trials=None):
        '''
        Initialize centers by kmeans++ algorithm.

        First assign a center randomly, then select the next center from several local candidates. The probabilities of candidates chosen 
        are according to their distances to existing centers. Among the local candidates, choose the one with minimum potential to be the 
        next center. Potential is the sum of distances from existing centers to all the samples.

        Parameters:
        ------------
        X : np.array, shape (n_samples, n_features)
        n_clusters : int
        n_local_trials : int, number of local candidates at each trial

        Returns:
        ------------
        centers : np.array, shape (n_clusters, n_features)
        '''
        n_samples, n_features = X.shape

        centers = np.zeros((n_clusters, n_features), dtype=X.dtype)
        
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))

        # Pick first center randomly
        center_id = np.random.randint(n_samples)
        centers[0] = X[center_id]
        closest_dist_square = self._euclidean_distance(centers[np.newaxis, 0], X)
        current_potential = closest_dist_square.sum() # \sum D(X) ** 2
        
        for c in range(1, n_clusters):
            rand_vals = np.random.random_sample(size=n_local_trials) * current_potential
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_square), rand_vals)
            distance_to_candidates = self._euclidean_distance(X[candidate_ids], X) # Shape (n_local_trials, n_samples)
            np.minimum(distance_to_candidates, closest_dist_square, out=distance_to_candidates)
            candidates_potential = distance_to_candidates.sum(axis=1)
            
            best_candidate = np.argmin(candidates_potential)
            current_potential = candidates_potential[best_candidate]
            closest_dist_square = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            centers[c] = X[best_candidate]
        return centers

    def _euclidean_distance(self, X, Y, squared=True):
        '''
        Calculate the euclidean distance between matrix X and Y.

        Parameters:
        ------------
        X : np.array, shape (n_samples1, n_features)
        Y : np.array, shape (n_samples2, n_features)

        Returns:
        ------------
        dist : np.array, shape (n_samples1, n_samples2)
        '''
        XX = (X * X).sum(axis=1)
        XX = XX[:, np.newaxis]  # (n_samples1, None)
        YY = (Y * Y).sum(axis=1) 
        YY = YY[np.newaxis, :] # (None, n_samples2)
        XY = X @ Y.T
        dist = XX + YY - 2 * XY
        if not squared:
            dist = np.sqrt(dist)
        return dist

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
    # X = np.array(
    #     [[1, 2], [1, 4], [1, 0],
    #     [10, 2], [10, 4], [10, 0]]
    # )
    X = np.random.randn(100, 30)
    kmeans = KMeans(n_clusters=2, max_iter=100, tol=1e-5, init='kmeans++').fit(X)
    print(kmeans.labels_, '\n', kmeans.cluster_centers_)
    
if __name__ == "__main__":
    test()
