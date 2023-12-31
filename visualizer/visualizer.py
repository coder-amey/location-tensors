#External imports
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

#Internal imports
from global_config.global_config import (
    NUM_CAMS, SCALE_DOWN_FACTOR, IMAGE_WIDTH, IMAGE_HEIGHT)


# IMPORTANT DEFINITIONS
    #Footage: a sequence of views
    #View: a snapshot of all pixel values
    
def trajectory2views(trajectory, obj_id, footage):
    for frame, [k, x1, y1, x2, y2] in enumerate(trajectory):
        # Re-index the cameras.
        k -= 1
        # Scale down the image size.
        x1, y1, x2, y2 = [co_ord // SCALE_DOWN_FACTOR for co_ord in  [x1, y1, x2, y2]]
        footage[frame][k][y1:y2, x1:x2] = obj_id
    return footage


def display_views(footage, num_cams=NUM_CAMS, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    cmap = colors.LinearSegmentedColormap.from_list('Custom Cmap', [plt.cm.Set3(i) for i in range(plt.cm.Set3.N)], plt.cm.Set3.N)
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
                ax[k//4][k%4].imshow(view[k], cmap="Set3", norm=colors.Normalize(vmin=0, vmax=11))      # Background colour = 0
                prev_view[k] = view[k]
        fig.suptitle(f"Frame {frame} / {num_frames} ({round(frame * 100 / num_frames, 2)}%)")
        plt.pause(0.005)
    print()


def project_trajectories(trajectories, num_cams=NUM_CAMS, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    footage = [np.zeros((num_cams, height, width)) for _ in range(max([len(trajectory) for trajectory in trajectories]))]
    for obj_index, trajectory in enumerate(trajectories):
        footage = trajectory2views(trajectory, obj_index + 1, footage)
        # print(f"\rProgress: {round(frame * 100 / num_frames, 2)}%", end="")
    display_views(footage)