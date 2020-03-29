import numpy as np
import sklearn.preprocessing

class MultinomialNB(object):
    '''
    Multi-nomial distribution Naive Bayes. It is suitable for discrete features.
    P(x_i | y_j) = (P(x_i y_j) + alpha) / (P(y_j) + alpha * n_features), 
    where alpha (smoothing prior) is used to handle when some feature, e.g. x_i does not present in the training set (zero possibilities).

    Parameters:
    -----------
    alpha : float, smoothing prior
    '''
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, sample_weights=None):
        '''
        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
        y : np.array, shape (n_samples,)
        sample_weights : np.array, shape (n_samples,)
        '''
        _, self.n_features = X.shape

        # Binarize labels to simplfy counting process.
        labelbin = sklearn.preprocessing.LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        n_effective_classes = Y.shape[1]
        
        # Condition with only one class. LabelBinarizer returns one number 0 or 1 (for each sample).
        if n_effective_classes == 1:
            Y = np.concatenate((1-Y, Y), axis=1) 
        
        if sample_weights is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weights = np.asarray(sample_weights, dtype=Y.dtype)
            Y *= sample_weights[:, np.newaxis]

        # Initialize feature_count and class_count
        self._init_counters(n_effective_classes=n_effective_classes, n_features=self.n_features)
        self._count(X, Y)
        self._update_feature_log_prob(feature_count=self.feature_count_, alpha=self.alpha)
        self._update_class_log_prior(class_count=self.class_count_)
        return self

    def predict(self, X):
        '''
        Parameters:
        ------------
        X : np.array, shape (n_samples, n_features)
        '''
        joint_log_likelihood = X @ self.feature_log_prob_.T + self.class_log_prior_ # shape (n_samples, n_effective_classes)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)

    def _count(self, X, Y):
        '''
        Count the features (of each class) and classes.

        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
        Y : np.array, shape (n_samples, n_effective_classes)

        Attributes:
        -----------
        class_count_ : np.array, shape (n_effective_classes,), samples number in classes
        feature_count : np.array, shape (n_effective_classes, n_features), feature occurences in classes.
        '''
        self.class_count_ += Y.sum(axis=0)
        self.feature_count_ += Y.T @ X
    
    def _update_feature_log_prob(self, feature_count, alpha):
        smooth_fc = feature_count + alpha # shape (n_effective_classes, n_features)
        smooth_cc = smooth_fc.sum(axis=1) # shape (n_effective_classes,)
        self.feature_log_prob_ = np.log(smooth_fc) - np.log(smooth_cc[:, np.newaxis]) # shape (n_effective_classes, n_features)

    def _update_class_log_prior(self, class_count):
        self.class_log_prior_ = np.log(class_count) - np.log(class_count.sum())

# ============================== Simple Test ============================================
def test():
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    clf = MultinomialNB().fit(X, y, sample_weights=list(range(1, 7)))
    print('Prediction:', clf.predict(X[2:3]))

if __name__ == '__main__':
    test()