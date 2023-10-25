gpu_server = False
parallel_objects = (not gpu_server) and False
selected_gpu = "0"

mode = "new"  # new or load
model_name = "prototype_giou_cce.ml"

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
    model, _, _ = lstm.train_model()
    lstm.save_model(model, name="prototype_giou_gpu.ml")

elif mode == "load":
    # Trained model
    model = lstm.load_model(name="prototype_giou_gpu.ml")


# Partially trained model
# model = lstm.load_model(name="customized_lstm.ml")
# model, _, _ = lstm.train_model(model=model)
# lstm.save_model(model, name="advanced_lstm.ml")

if not gpu_server:
    # Generate predictions
    from global_config.global_config import TENSOR_DATA_PATH
    from loader.ETL import tensor2trajectory
    from loader.loader import generate_tensors
    from utils.utils import load_pkl, tensor_decode_one_hot, tensor_encode_one_hot
    from visualizer.visualizer import project_trajectories

    import numpy as np
    import os

    obj_trajectories = load_pkl(os.path.join(TENSOR_DATA_PATH, "demo_trajectories.pkl"))
    
    if parallel_objects:
        Y = []
        Y_pred = []
        for key in ['10_13_12_106_10_13_13_190', '10_13_12_107_10_13_13_195']:
            # trajectory = list(obj_trajectories.keys())[key]
            x, y = generate_tensors(trajectories_dict={key: obj_trajectories[key]}, \
                 save_to_file=False)
            Y.append(y)
            Y_pred.append(lstm.predict(model, x, tensor_encode_one_hot(y)))

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
        y_pred, y_pred_encoded = lstm.predict(model, x, y_encoded, debug=True)

        # Plot predictions
        print(f"Output details: {y_pred_encoded.shape} -> {y_pred.shape} -> {len(tensor2trajectory(y_pred))}")
        print(f"Ground truth details: {y_encoded.shape} -> {y.shape} -> {len(tensor2trajectory(y))}")
        project_trajectories([tensor2trajectory(y), tensor2trajectory(y_pred)])
        
        # Debug
        print(f"Pred[0]:\n{y_pred[0, :, :]}")
        print(f"Pred_diff[0]:\n{y_pred[0, :, :] - y[0, :, :]}")
        print(f"Pred[-1]:{y_pred[-1, :, :]}")
        print(f"Pred_diff[-1]:\n{y_pred[-1, :, :] - y[-1, :, :]}")
