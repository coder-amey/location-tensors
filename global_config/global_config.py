import os

FEATURE_COLUMNS = ['camera', 'x1', 'y1', 'x2', 'y2']
DATA_PATH = os.path.join("/dcs/large/u2288122/Workspace/location-tensors/data/")
TRACKING_DATA_PATH = os.path.join(DATA_PATH, "tracking_data")
CSV_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "csv_data")
TENSOR_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "tensor_data")
N_INPUT_TSTEPS = 4
N_OUTPUT_TSTEPS = 12