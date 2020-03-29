import numpy as np

class PCA(object):
    '''
    Principal component analysis.

    SVD:
    X = USV' --> X'X = VS^2V' --> X'XV = S^2 V
    '''
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        '''
        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
        '''
        self._fit(X)
        return self

    def _fit(self, X):
        
        n_samples, _ = X.shape

        # Center data
        X -= X.mean(axis=0)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        
        self.components_ = V[:self.n_components]
        
        explained_variance_  = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_variance

        self.explained_variance_ = explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]

        return U, S, V
    
    def fit_transform(self, X):
        '''
        Parameters:
        ----------
        X : np.array, shape (n_samples, n_features)

        Returns:
        ----------
        U : np.array, shape (n_samples, n_components)
        '''
        U, S, V = self._fit(X)
        return U[:, :self.n_components] * S[np.newaxis, :self.n_components] 
    
# ============================= Simple Test =================================
def test():
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt

    data = load_iris().data
    data = np.asarray(data)
    print("Original Data Shape", data.shape)
    data_pca = PCA(n_components=2).fit_transform(data)
    print("PCA Data Shape", data_pca.shape)
    plt.plot(data_pca[:,0], data_pca[:,1], 'o')
    plt.show()

if __name__ == '__main__':
    test()