#External imports
import os
import numpy as np
import pandas as pd

#Internal imports
from utils.utils import (
    load_bbox_data, store_pkl, is_consecutive)
from global_config.global_config import (
    ALL_BOUNDING_BOXES_PATH, CROSS_CAM_MATCHES_PATH, ENT_DEP_PATH, CSV_DATA_PATH, TENSOR_DATA_PATH,
    COLUMNS, NUM_DAYS, FEATURE_COLUMNS,
    N_INPUT_TSTEPS, N_OUTPUT_TSTEPS)


# Trajectory-tensor interchangability functions
def trajectory2tensors(trajectory, n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS):
    window_len = n_input_tsteps + n_output_tsteps
    X = []
    Y = []
    for i in range(0, trajectory.shape[0] - window_len + 1):   # Right boundary is inclusive, hence +1.
        X.append(trajectory[i: i+n_input_tsteps])
        Y.append(trajectory[i+n_input_tsteps: i+window_len])
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def tensor2trajectory(tensor):
    trajectory = []
    for t_step in tensor[:-1]:
        trajectory.append(t_step[0])
    for t_step in tensor[-1]:
        trajectory.append(t_step)
    return(trajectory)


def bbs2trajectories(bbs_path=ALL_BOUNDING_BOXES_PATH, ent_dep_path=ENT_DEP_PATH, ccm_path=CROSS_CAM_MATCHES_PATH, save_to_files=True):
    """Loads the raw bounding box data and persists them into intermediate hourly datasets"""
    all_trajectories = pd.DataFrame()
    for day in range(1, NUM_DAYS):
        all_bounding_boxes = load_bbox_data(os.path.join(bbs_path, f"all_bounding_boxes_day_{day}.csv"))
        entrances_and_departures = load_bbox_data(os.path.join(ent_dep_path, f"entrances_and_departures_day_{day}.csv"))
        cross_camera_matches = load_bbox_data(os.path.join(ccm_path, f"day_{day}.csv"))

        set_id = 1
        # Process each hourly partition
        for bounding_boxes, ents_deps, cc_matches in zip(all_bounding_boxes, entrances_and_departures, cross_camera_matches):
            bounding_boxes.insert(1, "obj_id", "0")
            obj_id = 1
            # print("frame", "cam", "track", "next_cam", "next_frame", "next_track")
            for _, row in cc_matches.iterrows():
                frame, cam, track, next_cam, next_frame = row[['frame_num', 'camera', 'track', 'next_cam', 'next_cam_framenum']]
                entrance = ents_deps[(ents_deps["entrance"] == 1) & (ents_deps["camera"] == next_cam) & (ents_deps["frame_num"] == next_frame)]
                try:
                    assert entrance.shape[0] == 1
                    next_track = entrance["track"].item()
                    assert is_consecutive(bounding_boxes[(bounding_boxes["camera"] == cam) & (bounding_boxes["track"] == track)].frame_num.tolist())
                    assert is_consecutive(bounding_boxes[(bounding_boxes["camera"] == next_cam) & (bounding_boxes["track"] == next_track)].frame_num.tolist())
                
                    # print(frame, cam, track, next_cam, next_frame, next_track, sep="\t")
                    bounding_boxes.loc[(bounding_boxes["camera"] == cam) & (bounding_boxes["track"] == track), 'obj_id'] = f"{day}_{set_id}_{obj_id}"
                    bounding_boxes.loc[(bounding_boxes["camera"] == cam) & (bounding_boxes["track"] == track), 'obj_id'] = f"{day}_{set_id}_{obj_id}"
                    obj_id += 1
                
                except AssertionError:
                    print("Large occlusion was skipped.")

            bounding_boxes = bounding_boxes.loc[bounding_boxes["obj_id"] != "0", COLUMNS]
            
            if save_to_files:
                bounding_boxes.to_csv(os.path.join(CSV_DATA_PATH, f"day_{day}_set_{set_id}.csv"), index=False)

            all_trajectories = pd.concat([all_trajectories, bounding_boxes], ignore_index=True)
            set_id += 1
    
    # Separate each objects trajectory
    object_trajectories = {id: df[FEATURE_COLUMNS].to_numpy() for id, df in all_trajectories.groupby('obj_id')}
    
    if save_to_files:
        store_pkl(object_trajectories, os.path.join(TENSOR_DATA_PATH, "all_trajectories.pkl"))
    
    return object_trajectories


def generate_tensor_dataset(trajectories_dict, n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS):
    X = []
    Y = []
    for id, trajectory in trajectories_dict.items():
        if trajectory.shape[0] >= (n_input_tsteps + n_output_tsteps):
            x, y = trajectory2tensors(trajectory)
            X.append(x)
            Y.append(y)
    X = np.vstack(X)
    Y = np.vstack(Y)
    return (X, Y)