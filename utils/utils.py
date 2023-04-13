#External imports
import numpy as np
import os
import pandas as pd
import pickle

#Internal imports
from global_config.global_config import (
    CSV_DATA_PATH,
    FEATURE_COLUMNS,
    N_INPUT_TSTEPS, N_OUTPUT_TSTEPS)


# File handling functions:
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


# Trajectory-tensor interchangability functions
def trajectory2tensors(trajectory, n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS):
    window_len = n_input_tsteps + n_output_tsteps
    X = []
    Y = []
    for i in range(0, trajectory.shape[0] - window_len + 1):   #Right boundary is inclusive, hence +1.
        X.append(trajectory[i: i+n_input_tsteps])
        Y.append(trajectory[i+n_input_tsteps: i+window_len])
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def tensor2trajectory(tensor):
    count = 0
    trajectory = []
    for t_step in tensor[:-1]:
        trajectory.append(t_step[0])
        count += 1
    for t_step in tensor[-1]:
        trajectory.append(t_step)
        count += 1
    
    return(trajectory)




#Random
def compare(t1, t2):
    if len(t1) != len(t2):
        raise AttributeError(f"Unequal trajectories ({len(t1)} and {len(t2)})")
    else:
        for l1, l2 in zip(t1 , t2):
            if not np.array_equal(l1, l2):
                print(list(l1), list(l2), sep="\t")

