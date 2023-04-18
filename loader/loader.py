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
    # print(X.shape, Y.shape)

    key = list(trajectories_dict.keys())[0]
    orig_trajectory = trajectories_dict[key]
    orig_tensor_X, orig_tensor_Y = trajectory2tensors(orig_trajectory)
    total = len(orig_trajectory)
    print(f"Orig. Traj:\nlen: {total}")
    print(f"Orig. Tensors:\nX_len: {orig_tensor_X.shape}\tY_len: {orig_tensor_Y.shape}")
    print(total, total - N_OUTPUT_TSTEPS, total - N_INPUT_TSTEPS, sep="\t")
    reco_trajectory = tensor2trajectory(orig_tensor_Y)
    #print(*reco_trajectory, sep="\n")
    print(f"Reco. Traj:\nlen: {len(reco_trajectory)}")
    compare(trajectories_dict[key][-(total - N_INPUT_TSTEPS):], reco_trajectory)

    # Store the dataset
    if save_to_file:
        store_pkl(dataset, os.path.join(TENSOR_DATA_PATH, \
                f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl"))
    
    return dataset