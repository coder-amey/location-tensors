import os
import pandas as pd

COLUMNS = ['frame_num', 'camera', 'x1', 'y1', 'x2', 'y2', 'track']
DATA_PATH = os.path.join("/dcs/large/u2288122/Workspace/location-tensors/data/")
BOUNDING_BOXES_DATA_PATH = os.path.join(DATA_PATH, "bounding_box_data")
ALL_BOUNDING_BOXES_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "bounding_boxes")
CROSS_CAM_MATCHES_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "cross_camera_matches")
ENT_DEP_PATH = os.path.join(BOUNDING_BOXES_DATA_PATH, "entrances_and_departures")
TRACKING_DATA_PATH = os.path.join(DATA_PATH, "tracking_data")
OUTPUT_CSV_PATH = os.path.join(TRACKING_DATA_PATH, "csv_data")

def load_and_preprocess_data(file_loc):
    boxes_data = pd.read_csv(DATA_PATH + file_loc)

    # Sort by frame_num
    boxes_data = boxes_data.sort_values(by=['frame_num', 'camera', 'track'])
    
    # Partition on hours
    partitions = boxes_data.groupby('hour')
    partitions = [partitions.get_group(x).drop(['hour'], axis=1) for x in partitions.groups]
    return partitions

def is_consecutive(list):
    i = list[0]
    for j in list[1::]:
        if j - i != 1:
            return False

        i = j
    return True



if __name__ == '__main__':
    all_bounding_boxes = load_and_preprocess_data(os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_1.csv"))
    entrances_and_departures = load_and_preprocess_data(os.path.join(ENT_DEP_PATH, "entrances_and_departures_day1.csv"))
    cross_camera_matches = load_and_preprocess_data(os.path.join(CROSS_CAM_MATCHES_PATH, "day1.csv"))

    day = 1
    set_id = 1
    for bounding_boxes, ents_deps, cc_matches in zip(all_bounding_boxes, entrances_and_departures, cross_camera_matches):
        bounding_boxes.insert(-1, "obj_id", 0)
        obj_id = 1
        for _, row in cc_matches.iterrows():
            frame, cam, track, next_cam, next_frame = row[['frame_num', 'camera', 'track', 'next_cam', 'next_cam_framenum']]
            entrance = ents_deps[(ents_deps["entrance"] == 1) & (ents_deps["camera"] == next_cam) & (ents_deps["frame_num"] == next_frame)]
            assert entrance.shape[0] == 1
            next_track = entrance["track"]

            assert is_consecutive(bounding_boxes[(bounding_boxes["camera"] == cam) & (bounding_boxes["track"] == track)].frame_num.tolist())
            bounding_boxes.loc[(bounding_boxes["camera"] == cam) & (bounding_boxes["track"] == track), 'obj_id'] = f"{day}_{set_id}_{obj_id}"
            
            assert is_consecutive(bounding_boxes[(bounding_boxes["camera"] == next_cam) & (bounding_boxes["track"] == next_track)].frame_num.tolist())
            bounding_boxes.loc[(bounding_boxes["camera"] == cam) & (bounding_boxes["track"] == track), 'obj_id'] = f"{day}_{set_id}_{obj_id}"

            obj_id += 1

        bounding_boxes[bounding_boxes["obj_id"] != "0"].to_csv(os.path.join(OUTPUT_CSV_PATH, f"{day}_{set_id}.csv")
        