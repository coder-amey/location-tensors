gpu_server = True
parallel_objects = (not gpu_server) and False
selected_gpu = "0"

mode = "new"  # new, load or enhance
model_name = "mse_giou_diag_var_context_50ep.ml"
# mse_giou_diag_const_context_50ep.ml
# mse_giou_diag_var_context_50ep.ml

"""
CHECKLIST
=========
Set the following:
    gpu_server, mode, model_name    in run.py

    EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
	CAM_LOSS, BOX_LOSS, CAM_LOSS_WT, BOX_LOSS_WT    in global_config

If running on the GPU server, execute `module load cuda11.2`

Run this file
=========
"""

# Setup the server environment
if gpu_server:
    # Execute `module load cuda11.2`
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu

    import tensorflow as tf
    # tf.compat.v1.disable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                print(f"Using GPU: {gpu}")
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

# Create/load model
from model import custom_lstm as lstm

if mode == "new":
    # Untrained model
    model, logs = lstm.train_model()
    lstm.save_model(model, logs, name=model_name)

elif mode == "load":
    # Trained model
    model, logs = lstm.load_model(name=model_name)


# Partially trained model
if mode == "enhance":
    # Pre-trained model
    model, logs = lstm.load_model(name=model_name)
    model, new_logs = lstm.train_model(model)
    new_logs["train_log"] = {key: logs["train_log"][key] + new_logs["train_log"][key] for key in logs["train_log"].keys()}
    lstm.save_model(model, new_logs, name=f"enhanced_{model_name}")


if not gpu_server:
    # Generate predictions
    from global_config.global_config import TENSOR_DATA_PATH
    from loader.ETL import tensor2trajectory
    from loader.loader import generate_tensors
    from utils.utils import load_pkl, tensor_decode_one_hot, tensor_encode_one_hot
    from visualizer.visualizer import project_trajectories

    
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    print("Training curves:")
    cam_losses = [key for key in logs['train_log'].keys() if 'class' in key]
    box_losses = [key for key in logs['train_log'].keys() if 'reg' in key]

    plt.plot(logs['train_log']['loss'])
    plt.title(f'Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    for key in cam_losses:
        plt.plot(logs['train_log'][key])
    plt.title(f'Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    for key in box_losses:
        plt.plot(logs['train_log'][key])
    plt.title(f'Regression Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    obj_trajectories = load_pkl(os.path.join(TENSOR_DATA_PATH, "demo_trajectories.pkl"))
    
    if parallel_objects:
        Y = []
        Y_pred = []
        for key in ['10_13_12_106_10_13_13_190', '10_13_12_107_10_13_13_195']:
            # trajectory = list(obj_trajectories.keys())[key]
            x, y = generate_tensors(trajectories_dict={key: obj_trajectories[key]}, \
                 save_to_file=False)
            Y.append(y)
            Y_pred.append(lstm.predict(model, x))

        trajectories = []
        for y, y_pred in zip(Y, Y_pred):
            trajectories.append(tensor2trajectory(y))
            trajectories.append(tensor2trajectory(y_pred))
        project_trajectories(trajectories)

    else:
        # key = list(obj_trajectories.keys())['1_11_5_63']
        key = '10_13_12_106_10_13_13_190'
        x, y = generate_tensors(trajectories_dict={key: obj_trajectories[key]}, save_to_file=False)
        y_encoded = tensor_encode_one_hot(y)
        print(f"I/O shapes: ({x.shape}), ({y_encoded.shape})")
        y_pred = lstm.predict(model, x)

        # Plot predictions
        print(f"Output details: {y_pred.shape} -> {len(tensor2trajectory(y_pred))}")
        print(f"Ground truth details: {y_encoded.shape} -> {y.shape} -> {len(tensor2trajectory(y))}")
        project_trajectories([tensor2trajectory(y), tensor2trajectory(y_pred)])
        
        # Debug
        print(f"Pred[0]:\n{y_pred[0, :, :]}")
        print(f"Pred_diff[0]:\n{y_pred[0, :, :] - y[0, :, :]}")
        print(f"Pred[-1]:{y_pred[-1, :, :]}")
        print(f"Pred_diff[-1]:\n{y_pred[-1, :, :] - y[-1, :, :]}")
