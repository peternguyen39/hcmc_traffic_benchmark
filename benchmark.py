import numpy as np
import argparse
import json
import os

sensor_dict = None

def main():
    global sensor_dict
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        type=str, help='string to identify model')
    args = parser.parse_args()

    model_str = args.model
    if model_str not in ['ha', 'rfr', 'svr', 'ann', 'linear', 'linear-lasso', 'linear-ridge']:
        raise ValueError(
            'Select ha, rfr, svr, ann, linear, linear-lasso, or linear-ridge')
    graph_path = os.path.join(".",'graph', 'hcmc-clustered-graph.json')
    traffic_data_path = os.path.join('data', 'hcmc-traffic-data.json')
    training_dates = ['2022-04-05', '2022-04-06', '2022-04-07', '2022-04-08', '2022-04-09', '2022-04-10', '2022-04-11',
     '2022-04-12', '2022-04-13', '2022-04-14', '2022-04-15', '2022-04-16', '2022-04-17', '2022-04-18', '2022-04-19', '2022-04-20',
      '2022-04-21', '2022-04-22','2022-04-23', '2022-04-24', '2022-04-25', '2022-04-26', '2022-04-27', '2022-04-28', '2022-04-29',
       '2022-04-30', '2022-05-01', '2022-05-02', '2022-05-03', '2022-05-04'] 
    testing_dates=['2022-05-05', '2022-05-05_1', '2022-05-06', '2022-05-07', '2022-05-07_1', '2022-05-08', '2022-05-09']
    adj_matrix, dist_matrix, sensor_dict = load_traffic_graph(graph_path)
    training_data = load_traffic_data(traffic_data_path, dates=training_dates)
    testing_data = load_traffic_data(traffic_data_path, dates=testing_dates)

    training_neighborhood_dict,neighbors = load_K_hop_neighborhood(adj_matrix, 0, sensor_dict, 2)
    training_neighborhood_data = load_neighborhood_data(training_data, neighbors)
    interpolated_data,n_samples = interpolate_traffic_data(training_neighborhood_data, 5)
    # print(training_neighborhood_dict)
    # for i in list(training_neighborhood_data.keys()):
    #     print(training_neighborhood_data[i].keys())
    temp=load_training_data(training_dates,interpolated_data,training_neighborhood_dict,neighbors,0,0,n_samples=n_samples,normalize_max=True)
    print('Feature_vector:',temp[0].tolist())
    # print("Ground_truth:",temp[1].tolist())
    # print("Max_dict:",temp[2])
    # print(testing_neighborhood_dict)
    

    

def load_traffic_graph(path):
    with open(path, 'r',encoding='utf8') as f:
        graph = json.load(f)
    #return adjaceny marix, distance matrix, and sensor dictionary
    #Traffic graph format:
    # {'adjacency-matrix':adj_matrix,'distance-matrix':dist_matrix,'camera-dictionary':sensor_dict}
    adj_matrix = np.array(graph['adjacency-matrix'])
    dist_matrix = np.array(graph['distance-matrix'])
    sensor_dict = graph['camera-dictionary']
    return adj_matrix, dist_matrix, sensor_dict

def load_traffic_data(path,dates='all'):
    with open(path, 'r',encoding='utf8') as f:
        traffic_data = json.load(f)
    #Traffic data format:
    # {date:{sensor_id:{sample_file_name:{'count':count,'timestamp':[hour,minute]}}}}
    if dates == 'all':
        return traffic_data
    else:
        return {date:traffic_data[date] for date in dates}

#Load K-hop neighborhood of a sensor
#Return both the neighborhood dictionary and the list of neighbors
def load_K_hop_neighborhood(adj_matrix,sensor_id,sensor_dict,K):
    neighborhood_dict = {0:[]}
    neighborhood_dict[0].append(sensor_id)
    neighbors = [sensor_id]
    neighbor_list = [sensor_id]
    for k in range(1,K+1):
        neighborhood_dict[k] = []
        for neighbor in neighbors:
            neighborhood_dict[k].extend(i for i in np.where(adj_matrix[neighbor])[0])
            neighbor_list.extend(i for i in np.where(adj_matrix[neighbor])[0])
        neighbors = neighborhood_dict[k]
    for i in neighborhood_dict:
        neighborhood_dict[i] = list(dict.fromkeys(neighborhood_dict[i]))
        if i!=0:
            for j in range(i):
                neighborhood_dict[i] = list(set(neighborhood_dict[i])-set(neighborhood_dict[j]))
    return neighborhood_dict, list(dict.fromkeys(neighbor_list))

def load_neighborhood_data(traffic_data,neighbors):
    neighborhood_data = {}
    for date in traffic_data:
        neighborhood_data[date] = {}
        for sensor_id in neighbors:
            try:
                neighborhood_data[date][str(sensor_id)] = traffic_data[date][str(sensor_id)]
            except:
                print('Sensor {} not found in traffic data. Available keys:{}'.format(sensor_id,traffic_data[date].keys()))
                print("Key data type:{}, dict key data type:{}".format(type(sensor_id),type(list(traffic_data[date].keys())[0])))
    return neighborhood_data

def interpolate_traffic_data(traffic_data,interval):
    #TODO: VERIFY CORRECTNESS
    """
    Apply nearest neighbor interpolation to traffic data at a given interval
    Interpolated_data format:
    # {date:{sensor_id:{'count':count,'timestamp':[hour,minute]}}}
    {date:{sensor_id:{sample_number}={count':count,'timestamp':[hour,minute]}}}}
    """
    interpolated_data = {}
    max_hour = 23
    min_hour = 7
    n_samples = int(60*(max_hour-min_hour)/interval)
    for date in traffic_data:
        date_data = traffic_data[date]
        interpolated_data[date] = {}
        # Iterate through each sensor
        for sensor_id in date_data:
            sensor_data = date_data[sensor_id]
            interpolated_data[date][sensor_id] = {}
            timestamps, file_names = load_timestamps(sensor_data,min_hour)
            # Iterate through each sample file
            for i in range(n_samples):
                interpolated_data[date][sensor_id][i] = {}
                # Interpolate data using nearest neighboring timestamp
                # Tried using linear interpolation (np.interp) but results in floating point values
                # interpolated_data[date][sensor_id][i]['count'] = np.interp(i,timestamps,sensor_data[file_names]['count'])
                interpolated_data[date][sensor_id][i]['count'] = sensor_data[file_names[np.argmin(np.abs(timestamps-i))]]['count']
                interpolated_data[date][sensor_id][i]['timestamp'] = [min_hour+i//60,i%60]
    return interpolated_data, n_samples


def load_timestamps(sensor_data,min_hour):
    timestamps=[]
    names = []
    for name in sensor_data:
        timestamp = sensor_data[name]['timestamp']
        value = (timestamp[0]-min_hour)*60+timestamp[1]
        timestamps.append(value)
        names.append(name)
    return np.array(timestamps), np.array(names)


def build_feature_vector(idx,n_samples, date, neighborhood_data,neighborhood,history,use_time_of_day,max_dict):
    features = []
    if use_time_of_day:
        features.append([idx])
    if isinstance(history,int):
        history = [history]
    if isinstance(idx,int):
        idx = str(idx)
    # neighborhood = 
    for h in history:
        for k in neighborhood:
            for sensor in neighborhood[k]:
                if len(max_dict)>0:
                    if max_dict[str(sensor)]==0:
                        features.append([0])
                    else:
                        if str(sensor) not in neighborhood_data[date]:
                            print('Sensor {} not found in neighborhood data. Available keys: {}'.format(str(sensor),neighborhood_data[date].keys()))
                            features.append([0])
                        elif idx-h not in neighborhood_data[date][str(sensor)]:
                            print('Sensor {} not found in neighborhood data at time {}'.format(sensor,idx-h))
                            features.append([0])
                        else:
                            features.append([neighborhood_data[date][str(sensor)][idx-h]['count']/max_dict[str(sensor)]])
                else:
                    if sensor not in neighborhood_data[date] or idx-h not in neighborhood_data[date][str(sensor)]:
                        features.append([0])
                    else:
                        features.append([neighborhood_data[date][str(sensor)][idx-h]['count']])
    ret_features = [f for sublist in features for f in sublist]
    return np.array(ret_features)


def load_training_data(dates:list[str],neighborhood_data:dict,neighborhood:dict,neighbors:list,history:int,horizon:int,n_samples:int,use_time_of_day:bool=True,normalize_max:bool=False):
    """
    :param list[str] dates: list of dates used to load training data
    :param dict neighborhood_data: dictionary of traffic data, interpolated
    :param dict neighborhood: neighborhood dictionary, at each hop from 1-K
    :param list neighbors: list of all neighbors of the current sensor, within K hops
    :param int history: how far in the past should the data be trained on
    :param int horizon: how far ahead should the data be predicting
    :param int n_samples: number of total data points,
    :param bool use_time_of_day: use time of day as a feature for training
    """
    global sensor_dict
    #Training data format:
    # {date:{sensor_id:{'count':count,'timestamp':[hour,minute]}}}
    X = []
    y = []
    #Arange in the form of (start,stop)
    valid_sample_indices = np.arange(history,n_samples-horizon)
    home_sensor = neighbors[0]
    # print("Neighborhood_data:{}".format(neighborhood_data))
    print("Neighborhood_data keys:{}".format(neighborhood_data.keys()))
    print("Neighborhood:{}".format(neighborhood))
    print("Neighbors:{}".format(neighbors))
    max_dict = {}
    if normalize_max:
        for sensor in neighbors:
            max_dict[str(sensor)] = 0
            for date in dates:
                try:
                    for sample in neighborhood_data[date][str(sensor)]:
                        if neighborhood_data[date][str(sensor)][sample]['count']>max_dict[str(sensor)]:
                            max_dict[str(sensor)] = neighborhood_data[date][str(sensor)][sample]['count']
                except KeyError as e:
                    print("--------------------")
                    print("Date:{}".format(date))
                    print("Neighborhood_data keys:{}".format(neighborhood_data[date].keys()))
                    print("KeyError:{}".format(e))
                    print("Corresponding camera name:{}".format(sensor_dict[str(sensor)][1]))
                    print("Corresponding camera coordinates:{}".format(sensor_dict[str(sensor)][0]))
    print(max_dict)
    for date in dates:
        for sample in valid_sample_indices:
            X.append(build_feature_vector(sample,n_samples,date,neighborhood_data,neighborhood,history,use_time_of_day,max_dict))
            if len(max_dict)>0:
                if str(home_sensor) not in max_dict:
                    max_dict[str(home_sensor)] = 1
                y.append(neighborhood_data[date][str(home_sensor)][sample+horizon]['count']/max_dict[str(home_sensor)])
            else:
                y.append(neighborhood_data[date][str(home_sensor)][sample+horizon]['count'])
    return np.array(X), np.array(y), max_dict


if __name__ == "__main__":
    main()
