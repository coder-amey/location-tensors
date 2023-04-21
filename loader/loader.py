#External imports
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

#Internal imports
from utils.utils import (
    load_pkl, store_pkl, tensor_encode_one_hot)
from loader.ETL import (
    trajectory2tensors)
from global_config.global_config import (
    TENSOR_DATA_PATH, N_INPUT_TSTEPS, N_OUTPUT_TSTEPS,
    TRAIN_TEST_SPLIT, RANDOM_SEED)

'''
CAVEATS:
    Occlusion between cameras is not considered!
    Multi-camera detection treated as separate objects.
'''

#Important definitions:
"""
    Trajectory: a sequence (list) of locations of a bounding box corresponding to a single tracked object.
    Tensor: an array of location tensors of dimension: n_timesteps x input/output_steps x features
"""


def generate_tensors(trajectories_dict=None, pickle_path=os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"), \
                     n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS, save_to_file=True):    
    # Load trajectories from pickle-file if necessary
    if trajectories_dict is None:
        trajectories_dict = load_pkl(pickle_path)

    # Generate the tensors dataset from the trajectories
    X = []
    Y = []
    for id, trajectory in trajectories_dict.items():
        if trajectory.shape[0] >= (n_input_tsteps + n_output_tsteps):
            x, y = trajectory2tensors(trajectory)
            X.append(x)
            Y.append(y)
    dataset = (np.vstack(X), np.vstack(Y))
    X, Y = dataset
    print(f"Dataset (with dimensions: {X.shape}, {Y.shape}) generated successfully.") 

    # Store the dataset
    if save_to_file:
        store_pkl(dataset, os.path.join(TENSOR_DATA_PATH, \
                f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl"))
    
    return dataset


def load_dataset(dataset_path=os.path.join(TENSOR_DATA_PATH, f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl"),\
                    split_fraction=TRAIN_TEST_SPLIT, seed=RANDOM_SEED):
    # Load, split and return the training and testing data
    X, Y = load_pkl(dataset_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_fraction, random_state=seed)
    Y_train_encoded = tensor_encode_one_hot(Y_train)
    Y_test_encoded = tensor_encode_one_hot(Y_test)
    print(f"Dataset loaded and encoded successfully.")
    print(f"X_train: {X_train.shape}\tY_train (encoded):{Y_train.shape}\nX_test: {X_test.shape}\tY_test (encoded):{Y_test.shape}")
    return [tf.convert_to_tensor(array, dtype=tf.float32) for array in  [X_train, Y_train, Y_train_encoded, X_test, Y_test, Y_test_encoded]]