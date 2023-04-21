# Create/load model
from model import lstm

# model, _, _ = lstm.train_model()
# lstm.save_model(model, name="prototype.ml")
model = lstm.load_model(name="prototype.ml")

# Generate predictions
from global_config.global_config import TENSOR_DATA_PATH
from loader.ETL import tensor2trajectory
from loader.loader import generate_tensors
from utils.utils import load_pkl

import numpy as np
import os

obj_trajectories = load_pkl(os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"))
key = list(obj_trajectories.keys())[100]
x, y = generate_tensors(trajectories_dict={key: obj_trajectories[key]}, save_to_file=False)
y_encoded = lstm.tensor_encode_one_hot(y)
print(f"I/O shapes: ({x.shape}), ({y_encoded.shape})")
y_pred_encoded = lstm.predict(model, x, y_encoded)

# Plot predictions
from visualizer.visualizer import project_trajectories
y_pred = lstm.tensor_decode_one_hot(y_pred_encoded.astype(int))
print(f"Output details: {y_pred_encoded.shape} -> {y_pred.shape} -> {len(tensor2trajectory(y_pred))}")
print(f"Ground truth details: {y_encoded.shape} -> {y.shape} -> {len(tensor2trajectory(y))}")
project_trajectories([tensor2trajectory(y), tensor2trajectory(y_pred)])
print(y_pred[:, :, 1])
print(y[:, :, 1])
exit()


'''
x = np.expand_dims(X_test[5,:,:], axis=0)
y = np.expand_dims(Y_test_encoded[5,:,:], axis=0)
#TESTS


from loader.ETL import bbs2trajectories
# obj_trajectories = bbs2trajectories()
# dataset = generate_tensors(save_to_file=True)
# test = load_dataset()

from utils.utils import load_pkl
from global_config.global_config import TENSOR_DATA_PATH, N_INPUT_TSTEPS, N_OUTPUT_TSTEPS
from visualizer.visualizer import project_trajectories
import os
import pandas as pd
trajectories_df = load_pkl(os.path.join(TENSOR_DATA_PATH, "trajectories_df.pkl"))
obj_trajectories = load_pkl(os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"))
trajectories_df = trajectories_df.filter(lambda df: len(df) >= N_INPUT_TSTEPS + N_OUTPUT_TSTEPS).groupby('obj_id')

keys = list(obj_trajectories.keys())
# print(f"{keys[0]}:\n{obj_trajectories[keys[0]]}")
footage = project_trajectories([obj_trajectories[keys[i]] for i in range(50, 55)])'''