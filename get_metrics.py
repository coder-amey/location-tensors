model_name = "robust_lstm_mse_ep200.ml" # "robust_lstm.ml"

from model import custom_lstm as lstm
from global_config.global_config import N_INPUT_TSTEPS, N_OUTPUT_TSTEPS, TENSOR_DATA_PATH, MODEL_PATH
from utils.utils import load_pkl

import matplotlib.pyplot as plt
import os
import tensorflow as tf

def load_data(name=f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl", path=TENSOR_DATA_PATH):
    X, Y = load_pkl(os.path.join(path, name))
    return [tf.convert_to_tensor(array, dtype=tf.float32) for array in  [X, Y]]

print(f"Loading {model_name} model...")
model, logs = lstm.load_model(name=model_name)
for dataset_name in [f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl", "demo_dataset.pkl"]:
    print(f"Loading {dataset_name} dataset...")
    X, Y = load_data(name=dataset_name)
    print("Drawing metrics...")

    AP_cams, AP_parts, siou_score, ade, PR_cams, PR_parts = lstm.calculate_metrics(model, X, Y)
    print(f"Average Precision:\n\tCamera:\t{AP_cams}\n\tBoxes:\t{AP_parts}")
    print(f"SIoU score:\t{siou_score}")
    print(f"ADE:\t:{ade}")

    metrics_file = 'metrics.log'
    with open(os.path.join(MODEL_PATH, model_name, metrics_file), 'a+') as file:
        file.write(f"{dataset_name}:\n")
        file.write(f"Average Precision:\n\tCamera:\t{AP_cams}\n\tBoxes:\t{AP_parts}\n")
        file.write(f"SIoU score:\t{siou_score}\n\n")
        file.write(f"ADE:\t{ade}\n\n")

        file.write(f"PR_cams:\n{PR_cams}\n\n")
        file.write(f"PR_parts:\n{PR_parts}\n\n")

    plt.clf()
    plt.plot(PR_cams['precision'], PR_cams['recall'])
    plt.title(f'PR-curve for camera prediction')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    # plt.show()
    plt.savefig(os.path.join(MODEL_PATH, model_name, f"PR_cams_{dataset_name}.png"))

    plt.clf()
    plt.plot(PR_parts['precision'], PR_parts['recall'])
    plt.title(f'PR-curve for box prediction')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    # plt.show()
    plt.savefig(os.path.join(MODEL_PATH, model_name, f"PR_boxes_{dataset_name}.png"))
