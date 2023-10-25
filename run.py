gpu_server = False
selected_gpu = "0"
parallel_objects = True

# Setup the server environment
if gpu_server:
    # Execute `module load cuda11.2`
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu;

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

# Untrained model
# model, _, _ = lstm.train_model()
# lstm.save_model(model, name="prototype_2.ml")
model = lstm.load_model(name="advanced_lstm.ml")

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

    obj_trajectories = load_pkl(os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"))
    
    if parallel_objects:
        Y = []
        Y_pred = []
        for key in [47, 48]:
            trajectory = list(obj_trajectories.keys())[key]
            x, y = generate_tensors(trajectories_dict={trajectory: obj_trajectories[trajectory]}, save_to_file=False)
            Y.append(y)
            Y_pred.append(lstm.predict(model, x, tensor_encode_one_hot(y)))

        trajectories = []
        for y, y_pred in zip(Y, Y_pred):
            trajectories.append(tensor2trajectory(y))
            trajectories.append(tensor2trajectory(y_pred))
        project_trajectories(trajectories)

    else:
        key = list(obj_trajectories.keys())[47]
        x, y = generate_tensors(trajectories_dict={key: obj_trajectories[key]}, save_to_file=False)
        y_encoded = tensor_encode_one_hot(y)
        print(f"I/O shapes: ({x.shape}), ({y_encoded.shape})")
        y_pred, y_pred_encoded = lstm.predict(model, x, y_encoded, debug=True)

        # Plot predictions
        print(f"Output details: {y_pred_encoded.shape} -> {y_pred.shape} -> {len(tensor2trajectory(y_pred))}")
        print(f"Ground truth details: {y_encoded.shape} -> {y.shape} -> {len(tensor2trajectory(y))}")
        project_trajectories([tensor2trajectory(y), tensor2trajectory(y_pred)])
        
        # Debug
        # print(y_pred_encoded[:, :, 0:15])
        # print(y_pred[:, :, 0])
        # print(y[:, :, 0])
        # print(y[0, :, :])
        print(y_pred[0, :, :] - y[0, :, :])
        # print(y[-1, :, :])
        print(y_pred[-1, :, :] - y[-1, :, :])
    
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
