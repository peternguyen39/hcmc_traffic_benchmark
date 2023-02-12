import numpy as np
from sklearn.svm import SVR

class SupportVectorRegressor:
    def __init__(self, kernel='rbf'):
        self.kernel = kernel

    def fit(self, X, y):
        self.clf = SVR(kernel=self.kernel, gamma='scale')
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)