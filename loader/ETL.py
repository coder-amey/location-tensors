#External imports
import os
import pandas as pd

#Internal imports
from utils.utils import (
    load_bbox_data, is_consecutive)
from global_config.global_config import (
    ALL_BOUNDING_BOXES_PATH, CROSS_CAM_MATCHES_PATH, ENT_DEP_PATH, CSV_DATA_PATH,
    COLUMNS, OCCLUSION_THRESHOLD, NUM_DAYS)


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
            set_id += 1
            
            if save_to_files:
                bounding_boxes.to_csv(os.path.join(CSV_DATA_PATH, f"day_{day}_set_{set_id}.csv"), index=False)
            
            pd.concat(all_trajectories, bounding_boxes)
            
    return all_trajectories