import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ['frame_num', 'camera', 'x1', 'y1', 'x2', 'y2', 'track']
DATA_PATH = "/dcs/large/u2288122/Workspace/Multi-Camera-Trajectory-Forecasting/data/bounding_boxes/"
#Add cam here to skip rows
NUM_CAMS = 15
ORIGINAL_IMAGE_WIDTH = 1920
ORIGINAL_IMAGE_HEIGHT = 1200
SCALE_DOWN_FACTOR = 100
IMAGE_WIDTH = 1920 // SCALE_DOWN_FACTOR
IMAGE_HEIGHT = 1200 // SCALE_DOWN_FACTOR

def tensor_generator(bounding_boxes):
    frame = 1
    last_frame = bounding_boxes.frame_num.unique().tolist()[-1]
    while(frame <= last_frame):
        frame += 1
        current_boxes = bounding_boxes.loc[bounding_boxes.frame_num == frame, ['camera', 'x1', 'y1', 'x2', 'y2', 'track']].sort_values(by=["track", "camera"])
        yield current_boxes[['camera', 'x1', 'y1', 'x2', 'y2']].to_numpy(), current_boxes.track.tolist()


def tensor2view(tensor, object_ids, num_cams=NUM_CAMS, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    view = np.zeros((num_cams, height, width))
    for [k, x1, y1, x2, y2], object_id in zip(tensor, object_ids):
        view[k][y1:y2, x1:x2] = object_id
    return view


def load_and_preprocess_data(file_name):
    print(f"Loading bounding boxes from {file_name}...")
    boxes_data = pd.read_csv(DATA_PATH + file_name)

    # Re-index the cameras.
    boxes_data[['camera']] = boxes_data[['camera']] - 1

    # Scale down the image size.
    boxes_data[['x1', 'y1', 'x2', 'y2']] = boxes_data[['x1', 'y1', 'x2', 'y2']] // SCALE_DOWN_FACTOR

    # Sort by frame_num
    boxes_data = boxes_data.sort_values(by=['frame_num', 'camera', 'track'])
    
    # Partition on hours
    all_bounding_boxes = boxes_data.groupby('hour')
    all_bounding_boxes = [all_bounding_boxes.get_group(x).drop(['hour'], axis=1) for x in all_bounding_boxes.groups]
    return all_bounding_boxes
    

def display_views(footage, num_frames):
    num_axes_rows = (NUM_CAMS // 4)
    if (NUM_CAMS % 4) != 0:
        num_axes_rows += 1
    fig, ax = plt.subplots(num_axes_rows, 4, figsize=(4 * IMAGE_WIDTH, num_axes_rows * IMAGE_HEIGHT))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    prev_view = np.zeros((NUM_CAMS, IMAGE_HEIGHT, IMAGE_WIDTH))
    for frame, view in footage[600::]:
        for k in range(view.shape[0]):
            if not np.array_equal(view[k], prev_view[k]):
                ax[k//4][k%4].imshow(view[k])
                prev_view = view
        fig.suptitle(f"Frame {frame} / {num_frames} ({round(frame * 100 / num_frames, 2)}%)")
        plt.pause(0.001)
    print()

if __name__ == '__main__':
    all_bounding_boxes = load_and_preprocess_data("all_bounding_boxes_day_1.csv")
    footage = []
    for bounding_boxes in all_bounding_boxes[0:1]:
        num_frames = bounding_boxes.frame_num.unique().tolist()[-1]
        print(f"Loaded {num_frames} timesteps...")
        frame = 0
        prev_view = np.zeros((NUM_CAMS, IMAGE_HEIGHT, IMAGE_WIDTH))
        for tensor, ids in tensor_generator(bounding_boxes):
            view = tensor2view(tensor, ids)
            frame += 1
            if np.array_equal(view, prev_view):
                    continue
            else:
                footage.append((frame, view))
                print(f"\rProgress: {round(frame * 100 / num_frames, 2)}%", end="")
        print()
        display_views(footage, num_frames)
        #plt.close(fig)