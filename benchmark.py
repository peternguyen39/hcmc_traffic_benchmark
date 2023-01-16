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
    graph_path = os.path.join('graph', 'hcmc-clustered-graph.json')
    traffic_data_path = os.path.join('data', 'hcmc-traffic-data.json')
    training_dates = ['2022-04-05', '2022-04-06', '2022-04-07', '2022-04-08', '2022-04-09', '2022-04-10', '2022-04-11',
     '2022-04-12', '2022-04-13', '2022-04-14', '2022-04-15', '2022-04-16', '2022-04-17', '2022-04-18', '2022-04-19', '2022-04-20',
      '2022-04-21', '2022-04-22','2022-04-23', '2022-04-24', '2022-04-25', '2022-04-26', '2022-04-27', '2022-04-28', '2022-04-29',
       '2022-04-30', '2022-05-01', '2022-05-02', '2022-05-03', '2022-05-04'] 
    testing_dates=['2022-05-05', '2022-05-05_1', '2022-05-06', '2022-05-07', '2022-05-07_1', '2022-05-08', '2022-05-09']
    adj_matrix, dist_matrix, sensor_dict = load_traffic_graph(graph_path)
    

def load_traffic_graph(path):
    with open(path, 'r') as f:
        graph = json.load(f)
    #return adjaceny marix, distance matrix, and sensor dictionary
    #Traffic graph format:
    # {'adjacency-matrix':adj_matrix,'distance-matrix':dist_matrix,'camera-dictionary':sensor_dict}
    adj_matrix = np.array(graph['adjacency-matrix'])
    dist_matrix = np.array(graph['distance-matrix'])
    sensor_dict = graph['camera-dictionary']
    return adj_matrix, dist_matrix, sensor_dict

def load_traffic_data(path,dates='all'):
    with open(path, 'r') as f:
        traffic_data = json.load(f)
    #Traffic data format:
    # {date:{sensor_id:{sample_file_name:{'count':count,'timestamp':[hour,minute]}}}}
    if dates == 'all':
        return traffic_data
    else:
        return {date:traffic_data[date] for date in dates}

if __name__ == "__main__":
    main()