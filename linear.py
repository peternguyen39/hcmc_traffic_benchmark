from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np

class LinearRegressor:
    def __init__(self, estimator='linear', alpha=1,n_jobs=-1):
        if estimator not in ['linear', 'lasso', 'ridge']:
            raise ValueError('Please pass linear, lasso, or ridge as estimator name')
        self.estimator = estimator
        self.alpha = alpha
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.estimator == 'linear':
            self.clf = LinearRegression(n_jobs=self.n_jobs)
        elif self.estimator == 'lasso':
            self.clf = Lasso(alpha=self.alpha)
        else:
            self.clf = Ridge(alpha=self.alpha)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
    