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
def generate_targets(Y, num_cams=NUM_CAMS):
    """Y(batch_size, t_steps, cams+4)
        -> t_steps * [Y_cam(batch_size, cams), Y_box(batch_size, 4)]"""
    targets = []
    for tensor in tf.split(Y, num_or_size_splits=12, axis=1):
        tensor = tf.squeeze(tensor, axis=1)
        targets.append(tensor[:, :num_cams])
        targets.append(tensor[:, num_cams:])
    return targets


def targets2tensors(targets, num_cams=NUM_CAMS, reformat_targets=True):
    """t_steps * [Y_cam(batch_size, cams), Y_box(batch_size, 4)]
        -> Y(batch_size, t_steps, cams+4)"""
    Y = []
    for i in range(0, len(targets), 2):
        if reformat_targets:
            Y.append(tf.concat([targets[i], to_two_point_format(targets[i + 1])], axis=1))
        else:
            Y.append(tf.concat([targets[i], targets[i + 1]], axis=1))
    return tf.stack(Y, axis=1)


def tensor_encode_one_hot(tensor):
    one_hot_tensor = []
    for object in tensor:
        one_hot_tensor.append(
            tf.expand_dims(
                tf.concat(
                    [tf.one_hot(object[:, 0], depth=NUM_CAMS), object[:, 1:]], axis=1
                ),
            axis=0)
        )
    return tf.concat(one_hot_tensor, axis=0)


def tensor_decode_one_hot(one_hot_tensor):
    tensor = []
    for object in one_hot_tensor:
        tensor.append(
            tf.expand_dims(
                tf.concat(
                    [tf.cast(
                        tf.expand_dims(
                            tf.argmax(object[:, 0:NUM_CAMS], axis=1),
                            axis=-1), dtype=tf.float32),
                    object[:, NUM_CAMS:]], axis=1),
                axis=0)
            )
    return tf.concat(tensor, axis=0)


def decode_2d_one_hot(one_hot_tensor):
    cam_tensor = tf.cast(
        tf.expand_dims(
            tf.argmax(one_hot_tensor[:, 0:NUM_CAMS], axis=1), axis=-1), dtype=tf.float32)
    pos_tensor = one_hot_tensor[:, NUM_CAMS:]
    return tf.concat([cam_tensor, pos_tensor], axis=1)


def to_center_point_format(bbox):
    # Convert from (x1, y1, x2, y2) to (mid_x, mid_y, width, height)
    x1, y1, x2, y2 = tf.unstack(bbox, axis=-1)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return tf.cast(tf.cast(
        tf.stack([mid_x, mid_y, width, height], axis=-1), \
            dtype=tf.int32), dtype=tf.float32)

def to_two_point_format(bbox):
    # Convert from (mid_x, mid_y, width, height) to (x1, y1, x2, y2)
    mid_x, mid_y, width, height = tf.unstack(bbox, axis=-1)
    x1 = mid_x - width / 2
    y1 = mid_y - height / 2
    x2 = mid_x + width / 2
    y2 = mid_y + height / 2
    return tf.cast(tf.cast(
        tf.stack([x1, y1, x2, y2], axis=-1), \
            dtype=tf.int32), dtype=tf.float32)


# Additional utilities
def get_partition(x1, y1, x2, y2, part_width=384, part_height=240):
    """
        Divide the (1920x1200) image into a grid of 25 rectangles of (384x240).
        Return an integer representing the rectangle to which the box belongs.
    """
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    part_num = str(int(x // part_width)) + str(int(y // part_height))
    return int(part_num, 5) #5 comes from the grid comprising of (5x5) unique cells


def is_consecutive(list):
    i = list[0]
    for j in list[1::]:
        if j - i > OCCLUSION_THRESHOLD:
            return False
        i = j
    return True


def validate_box(box):
    x1, y1, x2, y2 = box
    if x1 < x2 and y1 < y2:
        return True
    else:
        return False


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


# Metrics
def calculate_SIoU(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])  # Left-most x-coordinate of the intersection
    y1 = min(box1[1], box2[1])  # Top-most y-coordinate of the intersection
    x2 = min(box1[2], box2[2])  # Right-most x-coordinate of the intersection
    y2 = max(box1[3], box2[3])  # Bottom-most y-coordinate of the intersection

    # Check if there's a valid intersection
    if validate_box([x1, y1, x2, y2]):
        intersection = area([x1, y1, x2, y2])
    else:
        return 0

    return intersection / (area(box1) + area(box2) - intersection)


def SIoU(y_true, y_pred):
    """
    Inputs: y_true, y_pred of shape (num_samples, num_features)
    Outputs: score (floating value)
    """
    scores = []
    for truth, pred in zip(y_true, y_pred):
        k_true, k_pred = truth[0], pred[0]
        box_true = truth[1:]
        box_pred = pred[1:]
        if (k_true == k_pred) and validate_box(box_true) and validate_box(box_pred):
            scores.append(calculate_SIoU(box_true, box_pred))
        else:
            scores.append(0)
    return np.array(scores).mean()