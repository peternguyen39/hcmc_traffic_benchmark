import numpy as np

class HistoricalAverage:
    def __init__(self,sensor_id,normalize_max=False):
        self.sensor = sensor_id
        self.normalize_max = normalize_max

    def fit(self,neighborhood_data,training_dates):
        pass
        