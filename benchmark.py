import argparse
from benchmark_utils import *
from tqdm import tqdm
import sys

sensor_dict = None

def main():
    global sensor_dict
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', required=True,
    #                     type=str, help='string to identify model')
    # args = parser.parse_args()

    # model_str = args.model

    sensors = [0]

    # if model_str not in ['ha', 'rfr', 'svr', 'ann', 'linear', 'linear-lasso', 'linear-ridge']:
    #     raise ValueError(
    #         'Select ha, rfr, svr, ann, linear, linear-lasso, or linear-ridge')
    
    graph_path = os.path.join(".",'graph', 'hcmc-clustered-graph.json')
    traffic_data_path = os.path.join('data', 'hcmc-traffic-data.json')
    training_dates = ['2022-04-05', '2022-04-06', '2022-04-07', '2022-04-08', '2022-04-09', '2022-04-10', '2022-04-11',
     '2022-04-12', '2022-04-13', '2022-04-14', '2022-04-15', '2022-04-16', '2022-04-17', '2022-04-18', '2022-04-19', '2022-04-20',
      '2022-04-21', '2022-04-22','2022-04-23', '2022-04-24', '2022-04-25', '2022-04-26', '2022-04-27', '2022-04-28', '2022-04-29',
       '2022-04-30', '2022-05-01', '2022-05-02', '2022-05-03', '2022-05-04'] 
    
    testing_dates=['2022-05-05', '2022-05-05_1', '2022-05-06', '2022-05-07', '2022-05-07_1', '2022-05-08', '2022-05-09']
    
    interval = 5
    num_trials = 10
    adj_matrix, dist_matrix, sensor_dict = load_traffic_graph(graph_path)
    result_dict = {}
    all_res ={}
    
    raw_res = {}
    all_raw_res={}
    for idx,model_type in enumerate(['ha', 'rfr', 'svr', 'ann', 'linear', 'linear-lasso', 'linear-ridge']):
        result_dict = {}
        print(f"Running {model_type}, {idx+1}/7")
        model_str = model_type
        with tqdm(total = len(sensors)*len(testing_dates)*3*2*3*num_trials,file=sys.stdout) as pbar:
            for sensor in sensors:
                for test_date in testing_dates:
                    for K in range(3):
                        neighborhood_dict, neighbors = load_K_hop_neighborhood(adj_matrix, sensor, sensor_dict, K)
                        traffic_data = load_traffic_data(traffic_data_path, dates=training_dates+testing_dates)
                        neighborhood_data = load_neighborhood_data(traffic_data, neighbors)
                        interpolated_data, n_samples = interpolate_traffic_data(neighborhood_data, interval)
                        for history in [3,6]:
                            for horizon in [3,6,9]:
                                result_dict[(sensor, test_date, K, history, horizon)] = []
                                X, y, max_dict = load_training_data(training_dates, interpolated_data, neighborhood_dict, neighbors, history, horizon, n_samples=n_samples, normalize_max=False)
                                X_test,y_test,indices = load_testing_data(test_date, interpolated_data, neighborhood_dict, neighbors, history, horizon, n_samples=n_samples, max_dict=max_dict)

                                model = load_model(model_str,sensor,X.shape[1], history)
                                for i in range(num_trials):
                                    pbar.set_description(f"Sensor: {sensor}, Test Date: {test_date}, K: {K}, History: {history}, Horizon: {horizon}, Trial: {i}")
                                    if model_str == 'ha':
                                        model.fit(interpolated_data,training_dates)
                                        y_pred = model.predict(indices)
                                    else:
                                        model.fit(X, y)
                                        y_pred = model.predict(X_test)
                                    res = mae(y_test, y_pred)
                                    result_dict[(sensor, test_date, K, history, horizon)].append(res)
                                    raw_res[(sensor, test_date, K, history, horizon)] = [y_test.tolist(),y_pred.tolist()]
                                    tqdm.write(f"Sensor: {sensor}, Test Date: {test_date}, K: {K}, History: {history}, Horizon: {horizon}, Trial: {i}, MAE: {res}")
                                    pbar.update(1)
        print(result_dict)
        all_res[model_str] = result_dict.copy()
        all_raw_res[model_str] = raw_res.copy()

    print("All results:",all_res)
    
    temp_dict = {}
    for i in all_res:
        temp_dict[i] = {}
        for j in all_res[i]:
            temp_dict[i][str(j)] = all_res[i][j]
    with open('all_results.json', 'w',encoding='utf8') as fp:
        json.dump(temp_dict, fp,indent=4,ensure_ascii=False)

    temp_dict = {}
    for i in all_raw_res:
        temp_dict[i] = {}
        for j in all_raw_res[i]:
            temp_dict[i][str(j)] = all_raw_res[i][j]
    with open('all_raw_results.json', 'w',encoding='utf8') as fp:
        json.dump(temp_dict, fp,indent=4,ensure_ascii=False)
    



if __name__ == "__main__":
    main()
