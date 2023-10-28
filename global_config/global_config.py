import os
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow_addons.losses import GIoULoss

# PATHS
DATA_PATH = os.path.join("/dcs/large/u2288122/Workspace/location-tensors/data/")
TRACKING_DATA_PATH = os.path.join(DATA_PATH, "tracking_data")
BOUNDING_BOXES_DATA_PATH = os.path.join(DATA_PATH, "bounding_box_data")
CSV_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "csv_data")
TENSOR_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "tensor_data")
MODEL_PATH = os.path.join(DATA_PATH, "saved_models")
ALL_BOUNDING_BOXES_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "bounding_boxes")
CROSS_CAM_MATCHES_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "cross_camera_matches")
ENT_DEP_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "entrances_and_departures")
DEMO_DATA_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "demo_dataset")

# DATASET PARAMS
COLUMNS = ['frame_num', 'obj_id', 'camera', 'x1', 'y1', 'x2', 'y2']
FEATURE_COLUMNS = ['camera', 'x1', 'y1', 'x2', 'y2']
T_MARGIN = 5        # Frame delta from average_transition delay
MODE = "Load"   # Generate OR Load

# MODEL PARAMS
N_INPUT_TSTEPS = 4
N_OUTPUT_TSTEPS = 12
NUM_DAYS = 21
OCCLUSION_THRESHOLD = 6
RANDOM_SEED = 47
TRAIN_TEST_SPLIT = 0.25
NUM_FEATURES = 5
EPOCHS = 20
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
CAM_LOSS = CategoricalCrossentropy()
BOX_LOSS = GIoULoss()           # GIoULoss() or MeanSquaredError()
CAM_LOSS_WT = 100        # Needs motivation
BOX_LOSS_WT = 400        # MSE ~ 500000 -> 0.001; GIOU ~ 2 -> 400

# IMAGE-DATA PARAMS
NUM_CAMS = 15
ORIGINAL_IMAGE_WIDTH = 1920
ORIGINAL_IMAGE_HEIGHT = 1200
SCALE_DOWN_FACTOR = 25
IMAGE_WIDTH = 1920 // SCALE_DOWN_FACTOR
IMAGE_HEIGHT = 1200 // SCALE_DOWN_FACTOR