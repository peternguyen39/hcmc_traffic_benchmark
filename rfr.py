from sklearn.ensemble import RandomForestRegressor as RFR

class RandomForestRegressor:
    def __init__(self,n_estimators=50,criterion='absolute_error',n_jobs=-1) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.n_jobs = n_jobs

    def fit(self,X,y):
        self.clf = RFR(n_estimators=self.n_estimators,criterion=self.criterion,n_jobs=self.n_jobs)
        self.clf.fit(X,y)

    def predict(self,X):
        return self.clf.predict(X)
        
