import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

COLUMNS = ['frame_num', 'obj_id', 'camera', 'x1', 'y1', 'x2', 'y2']
DATA_PATH = os.path.join("/dcs/large/u2288122/Workspace/location-tensors/data/")
TRACKING_DATA_PATH = os.path.join(DATA_PATH, "tracking_data")
INPUT_CSV_PATH = os.path.join(TRACKING_DATA_PATH, "csv_data")
#Add cam here to skip rows
NUM_CAMS = 15
ORIGINAL_IMAGE_WIDTH = 1920
ORIGINAL_IMAGE_HEIGHT = 1200
SCALE_DOWN_FACTOR = 100
IMAGE_WIDTH = 1920 // SCALE_DOWN_FACTOR
IMAGE_HEIGHT = 1200 // SCALE_DOWN_FACTOR

def tensor_generator(bounding_boxes):
    frame = bounding_boxes.frame_num.unique().tolist()[0]
    last_frame = bounding_boxes.frame_num.unique().tolist()[-1]
    while(frame <= last_frame):
        frame += 1
        current_boxes = bounding_boxes.loc[bounding_boxes.frame_num == frame, ['obj_id', 'camera', 'x1', 'y1', 'x2', 'y2']].sort_values(by=["obj_id", "camera"])
        yield current_boxes[['camera', 'x1', 'y1', 'x2', 'y2']].to_numpy(), current_boxes.obj_id.tolist()


def tensor2view(tensor, object_ids, num_cams=NUM_CAMS, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    view = np.zeros((num_cams, height, width))
    for [k, x1, y1, x2, y2], object_id in zip(tensor, object_ids):
        view[k][y1:y2, x1:x2] = object_id
    return view


def load_and_preprocess_data(file_name):
    print(f"Loading bounding boxes from {file_name}...")
    boxes_data = pd.read_csv(os.path.join(INPUT_CSV_PATH, file_name))

    # Re-index the cameras.
    boxes_data[['camera']] = boxes_data[['camera']] - 1

    # Scale down the image size.
    boxes_data[['x1', 'y1', 'x2', 'y2']] = boxes_data[['x1', 'y1', 'x2', 'y2']] // SCALE_DOWN_FACTOR

    # Sort by frame_num
    boxes_data = boxes_data.sort_values(by=['frame_num', 'camera'])
    
    return boxes_data
    

def display_views(footage, num_frames):
    num_axes_rows = (NUM_CAMS // 4)
    if (NUM_CAMS % 4) != 0:
        num_axes_rows += 1
    fig, ax = plt.subplots(num_axes_rows, 4, figsize=(4 * IMAGE_WIDTH, num_axes_rows * IMAGE_HEIGHT))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    prev_view = np.zeros((NUM_CAMS, IMAGE_HEIGHT, IMAGE_WIDTH))
    for frame, view in footage:
        for k in range(view.shape[0]):
            if not np.array_equal(view[k], prev_view[k]):
                ax[k//4][k%4].imshow(view[k])
                prev_view = view
        fig.suptitle(f"Frame {frame} / {num_frames} ({round(frame * 100 / num_frames, 2)}%)")
        plt.pause(0.001)
    print()


def project_trajectory(tensors, ids):
    frame = 0
    for tensor, obj_ids in zip(tensors, ids):
        view = tensor2view(tensor, obj_ids)
        frame += 1
        footage.append((frame, view))
        print(f"\rProgress: {round(frame * 100 / num_frames, 2)}%", end="")
    print()
    display_views(footage, num_frames)
    

if __name__ == '__main__':
    bounding_boxes = load_and_preprocess_data("day_1_set_1.csv")
    footage = []
    num_frames = bounding_boxes.frame_num.unique().tolist()[-1]
    print(f"Loaded {num_frames} timesteps...")
    prev_view = np.zeros((NUM_CAMS, IMAGE_HEIGHT, IMAGE_WIDTH))
    tensors = []
    ids = []
    for tensor, obj_ids in tensor_generator(bounding_boxes):
        tensors.append(tensor)
        ids.append(obj_ids)
    
    project_trajectory(tensors, ids)
# TO-DO: distinguish trajectory and tensor
# TO-DO: def project_trajectory
