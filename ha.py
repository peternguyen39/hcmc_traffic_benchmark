import numpy as np

class HistoricalAverage:
    def __init__(self,sensor_id,normalize_max=False):
        self.sensor = str(sensor_id)
        self.normalize_max = normalize_max

    def fit(self,neighborhood_data,training_dates):
        self.ha_count={}
        values=[]
        for date in training_dates:
            date_sensor_dict = neighborhood_data[date][self.sensor]
            for sample in date_sensor_dict:
                entry = date_sensor_dict[sample]
                count = entry['count']
                values.append(count)
                if sample not in self.ha_count:
                    self.ha_count[sample]=[]
                self.ha_count[sample].append(count)
        max_value = np.max(values)
        for sample in self.ha_count:
            if self.normalize_max:
                self.ha_count[sample] = np.mean(np.array(self.ha_count[sample])/max_value)
            else:
                self.ha_count[sample] = np.mean(self.ha_count[sample])
        return
    def predict(self,indices):
        predictions = []
        for i in range(len(indices)):
            idx = indices[i]
            if idx in self.ha_count:
                predictions.append(self.ha_count[idx])
            else:
                predictions.append(0)
        return predictions
