from sklearn.ensemble import RandomForestRegressor as RFR

class RandomForestRegressor:
    def __init__(self,n_estimators=50,criterion='mae') -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion

    def fit(self,X,y):
        self.clf = RFR(n_estimators=self.n_estimators,criterion=self.criterion)
        self.clf.fit(X,y)

    def predict(self,X):
        return self.clf.predict(X)
        
