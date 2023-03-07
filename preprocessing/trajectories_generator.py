import numpy as np
import os
import pandas as pd
import pickle

FEATURE_COLUMNS = ['camera', 'x1', 'y1', 'x2', 'y2']
DATA_PATH = os.path.join("/dcs/large/u2288122/Workspace/location-tensors/data/")
TRACKING_DATA_PATH = os.path.join(DATA_PATH, "tracking_data")
CSV_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "csv_data")
TENSOR_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "tensor_data")
N_INPUT_TSTEPS = 4
N_OUTPUT_TSTEPS = 12

'''
CAVEATS:
    Occlusion between cameras is not considered!
    Multi-camera detection treated as separate objects.
'''

def trajectory2tensors(trajectory, n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS):
    window_len = n_input_tsteps + n_output_tsteps
    X = []
    Y = []
    for i in range(0, trajectory.shape[0] - window_len):
        X.append(trajectory[i: i+n_input_tsteps])
        Y.append(trajectory[i+n_input_tsteps: i+window_len])
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# TO-DO: tensor2trajectory

def load_trajectories_csv(data_path=CSV_DATA_PATH):
    # Concatenate the trajectories of all objects across days and sets.
    all_trajectories = pd.concat([
        pd.read_csv(os.path.join(data_path, file_name)) \
            for file_name in os.listdir(data_path) \
                if file_name.endswith('.csv')
    ])

    # Separate each objects trajectory
    object_trajectories = all_trajectories.groupby('obj_id')
    return {id: df[FEATURE_COLUMNS].to_numpy() for id, df in object_trajectories}

def store_pkl(object, output_file):
    try:
        with open(output_file, 'wb') as pkl_file:
            pickle.dump(object, pkl_file)
    except:
        print("Error storing data pickle.")
        raise

def load_pkl(input_file):
    try:
        with open(input_file, 'rb') as pkl_file:
            return pickle.load(pkl_file)
    except:
        print("Error storing data pickle.")
        raise

def generate_tensor_dataset(trajectories_dict, n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS):
    X = []
    Y = []
    for id, trajectory in trajectories_dict.items():
        if trajectory.shape[0] >= (N_INPUT_TSTEPS + N_OUTPUT_TSTEPS):
            x, y = trajectory2tensors(trajectory)
            X.append(x)
            Y.append(y)
    X = np.vstack(X)
    Y = np.vstack(Y)
    return (X, Y)

if __name__ == '__main__':
    # 1-time operation to create trajectories_dict file
    trajectories_dict = load_trajectories_csv()
    store_pkl(trajectories_dict, os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"))
    
    # Load trajectories from pickle-file
    trajectories_dict = load_pkl(os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"))

    # Generate the tensors dataset from the trajectories
    dataset = generate_tensor_dataset(trajectories_dict)
    X, Y = dataset
    print(X.shape, Y.shape)

    # Store the dataset
    store_pkl(dataset, os.path.join(TENSOR_DATA_PATH, \
                                    f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl"))

    