import os

# PATHS
DATA_PATH = os.path.join("/dcs/large/u2288122/Workspace/location-tensors/data/")
TRACKING_DATA_PATH = os.path.join(DATA_PATH, "tracking_data")
BOUNDING_BOXES_DATA_PATH = os.path.join(DATA_PATH, "bounding_box_data")
CSV_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "csv_data")
TENSOR_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "tensor_data")
ALL_BOUNDING_BOXES_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "bounding_boxes")
CROSS_CAM_MATCHES_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "cross_camera_matches")
ENT_DEP_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "entrances_and_departures")

# DATASET PARAMS
COLUMNS = ['frame_num', 'obj_id', 'camera', 'x1', 'y1', 'x2', 'y2']
FEATURE_COLUMNS = ['camera', 'x1', 'y1', 'x2', 'y2']

# MODEL PARAMS
N_INPUT_TSTEPS = 4
N_OUTPUT_TSTEPS = 12
NUM_DAYS = 21
OCCLUSION_THRESHOLD = 6
