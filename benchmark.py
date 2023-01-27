import numpy as np
import argparse
import json
import os


def main():
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

    testing_neighborhood_dict,neighbors = load_K_hop_neighborhood(adj_matrix, 0, sensor_dict, 2)
    testing_neighborhood_data = load_neighborhood_data(testing_data, neighbors)
    interpolated_data = interpolate_traffic_data(testing_neighborhood_data, 5)
    print(interpolated_data)
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
        try:
            for sensor_id in neighbors:
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


def load_training_data(dates,neighborhood_data,neighborhood,neighbors,history,horizon,n_samples,use_time_of_day=True,normalize_max=False):
    #Training data format:
    # {date:{sensor_id:{'count':count,'timestamp':[hour,minute]}}}
    X = []
    y = []
    for date in dates:
        pass

if __name__ == "__main__":
    main()
