#External imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

#Internal imports
from utils.utils import (
    load_pkl, store_pkl, compare)
from loader.ETL import (
    generate_tensor_dataset, trajectory2tensors, tensor2trajectory)
from global_config.global_config import (
    NUM_CAMS, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT, SCALE_DOWN_FACTOR, IMAGE_WIDTH, IMAGE_HEIGHT)


# IMPORTANT DEFINITIONS
    #Footage: a sequence of views
    #View: a snapshot of all pixel values
    
def trajectory2views(trajectory, obj_id, footage):
    for frame, [k, x1, y1, x2, y2] in enumerate(trajectory):
        footage[frame][k][y1:y2, x1:x2] = obj_id
    return footage


def display_views(footage, num_cams=NUM_CAMS, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    num_axes_rows = (num_cams // 4)
    if (num_cams % 4) != 0:
        num_axes_rows += 1
    fig, ax = plt.subplots(num_axes_rows, 4, figsize=(4 * width, num_axes_rows * height))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    prev_view = np.zeros((num_cams, height, width))
    num_frames = len(footage)
    for frame, view in enumerate(footage):
        for k in range(num_cams):
            if not np.array_equal(view[k], prev_view[k]):
                ax[k//4][k%4].imshow(view[k])
                prev_view[k] = view[k]
        # fig.suptitle(f"Frame {frame} / {num_frames} ({round(frame * 100 / num_frames, 2)}%)")
        plt.pause(0.001)
    print()


def project_trajectories(trajectories, num_cams=NUM_CAMS, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    view = np.zeros((num_cams, height, width))
    footage = [view] * max([len(trajectory) for trajectory in trajectories])
    for obj_index, trajectory in enumerate(trajectories):
        footage = trajectory2views(trajectory, obj_index + 1, footage)
        # print(f"\rProgress: {round(frame * 100 / num_frames, 2)}%", end="")
    display_views(footage)