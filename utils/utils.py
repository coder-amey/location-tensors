#External imports
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

#Internal imports
from global_config.global_config import (
    CSV_DATA_PATH,
    FEATURE_COLUMNS, OCCLUSION_THRESHOLD, NUM_CAMS)


# File handling functions:
def load_bbox_data(file_loc):
    boxes_data = pd.read_csv(file_loc)

    #Sort by frame_num
    boxes_data = boxes_data.sort_values(by=['frame_num', 'camera', 'track'])
    
    #Partition on hours
    partitions = boxes_data.groupby('hour')
    partitions = [partitions.get_group(x).drop(['hour'], axis=1) for x in partitions.groups]
    return partitions


def load_all_trajectories(data_path=CSV_DATA_PATH):
    """Loads all trajectories persisted using the ETL.bbs2trajectories function."""
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


# Tensor-manipulation functions
def tensor_encode_one_hot(tensor):
    one_hot_tensor = np.vstack([    \
        np.expand_dims( \
            np.hstack([ \
                tf.one_hot(example[:, 0], depth=NUM_CAMS), example[:, 1:]] \
            ), axis=0)
                for example in tensor])
    return one_hot_tensor


def tensor_decode_one_hot(one_hot_tensor):
    tensor = np.vstack([    \
        np.expand_dims( \
            np.hstack([ \
                np.expand_dims(np.argmax(example[:, 0:NUM_CAMS], axis=1), axis=1)   \
                , example[:, NUM_CAMS:]] \
            ), axis=0)
                for example in one_hot_tensor])
    return tensor


# Additional utilities
def is_consecutive(list):
    i = list[0]
    for j in list[1::]:
        if j - i > OCCLUSION_THRESHOLD:
            return False
        i = j
    return True

