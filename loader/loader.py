#External imports
import numpy as np
import os
import pandas as pd

#Internal imports
from utils.utils import (
    load_pkl, store_pkl, compare)
from loader.ETL import (
    generate_tensor_dataset, trajectory2tensors, tensor2trajectory)
from global_config.global_config import (
    TENSOR_DATA_PATH, N_INPUT_TSTEPS, N_OUTPUT_TSTEPS)

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


def load_dataset(trajectories_dict=None, save_to_file=True):    
    # Load trajectories from pickle-file if necessary
    if trajectories_dict is None:
        trajectories_dict = load_pkl(os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"))

    # Generate the tensors dataset from the trajectories
    dataset = generate_tensor_dataset(trajectories_dict)
    X, Y = dataset
    print(f"Dataset (with dimensions: {X.shape}, {Y.shape}) loaded successfully.") 

    # Store the dataset
    if save_to_file:
        store_pkl(dataset, os.path.join(TENSOR_DATA_PATH, \
                f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl"))
    
    return dataset