import numpy as np 

class LinearRegression(object):
    '''
    A simplified linear regression model modified from sklearn.

    This implementation is only used for the author's own study. No commercial use.
    '''
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weights=None):
        '''
        Fit linear model.

        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
        y : np.array, shape (n_samples,) or (n_samples, n_targets)
        sample_weights : np.array, shape (n_samples,)

        Attributes:
        ------------
        coef_ : np.array, shape (n_features,) or (n_features, n_targets), coefficient matrix.
        intercept_ : float, Independent term in the linear model.
        '''
        X, y, X_offset, y_offset = self._preprocess_data(X, y, self.fit_intercept, sample_weights=sample_weights)
        
   
        self.coef_ = self.lstsq(X, y)
        self._set_intercept(X_offset, y_offset, self.fit_intercept)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def _preprocess_data(self, X, y, fit_intercept, copy=True, sample_weights=None):
        '''
        If fit intercept, this function will center data (X, y) to have zero mean along axis 0: X = X - X_offset, y = y - y_offset.
        '''
        n_samples = X.shape[0]
        if sample_weights is None:
            sample_weights = np.ones(n_samples, dtype=X.dtype)

        X = X.copy(order='K') if copy else X
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=X.dtype)

        if fit_intercept:
            X_offset = np.average(X, axis=0, weights=sample_weights)
            X -= X_offset
            y_offset = np.average(y, axis=0, weights=sample_weights)
            y -= y_offset
        else:
            X_offset = np.zeros(X.shape[0], dtype=X.dtype)
            y_offset = X.dtype.type(0)
        return X, y, X_offset, y_offset

    def _set_intercept(self, X_offset, y_offset, fit_intercept):
        '''
        x A + b = B --> b = B - x A
        '''
        if fit_intercept:
            self.intercept_ = y_offset - X_offset @ self.coef_
        else:
            self.intercept_ = 0.

    def lstsq(self, A, B):
        '''
        LeaST-SQuares solution to a linear matrix equation A x = B. Compute x to minimize the squared Euclidean 2-norm: |(B-Ax)|^2_2

        Proof: grad((A x - B)^2) = 0 --> 2 * A' * (B - Ax) = 0 -->  x = (A'A)^{-1} A'B

        Parameters:
        -----------
        A : np.array, shape (M, N), Coefficient Matrix.
        B : np.array, shape (M,) or (M, K), Dependent Variable values.

        Returns:
        -----------
        x : np.array, shape (N,) or (N, K)
        '''
        m, n = A.shape
        if m != B.shape[0]:
            raise ValueError('Matrix shape not compatible.')
        # if B.ndim == 1:
        #     B = B[:, np.newaxis]
        x = np.linalg.inv(A.T @ A) @ A.T @ B
        return x
    
    def score(self, X, y, sample_weights=None):
        pass
# =============================== Simple Test ===============================================
def test():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([[1, 2, 3], [4, 5, 6]])) + 10
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    print(reg.coef_, '\n', reg.intercept_)
    print(reg.predict(X))
    print(y)

if __name__ == '__main__':
    test()