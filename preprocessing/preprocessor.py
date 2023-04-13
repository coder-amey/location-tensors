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
OCCLUSION_THRESHOLD = 6
NUM_DAYS = 21

def load_and_preprocess_data(file_loc):
    boxes_data = pd.read_csv(file_loc)

    # Sort by frame_num
    boxes_data = boxes_data.sort_values(by=['frame_num', 'camera', 'track'])
    
    # Partition on hours
    partitions = boxes_data.groupby('hour')
    partitions = [partitions.get_group(x).drop(['hour'], axis=1) for x in partitions.groups]
    return partitions


def is_consecutive(list):
    i = list[0]
    for j in list[1::]:
        if j - i > OCCLUSION_THRESHOLD:
            return False
        i = j
    return True


if __name__ == '__main__':
    for day in range(1, NUM_DAYS):
        all_bounding_boxes = load_and_preprocess_data(os.path.join(ALL_BOUNDING_BOXES_PATH, f"all_bounding_boxes_day_{day}.csv"))
        entrances_and_departures = load_and_preprocess_data(os.path.join(ENT_DEP_PATH, f"entrances_and_departures_day_{day}.csv"))
        cross_camera_matches = load_and_preprocess_data(os.path.join(CROSS_CAM_MATCHES_PATH, f"day_{day}.csv"))

        set_id = 1
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

            bounding_boxes = bounding_boxes[['frame_num', 'obj_id', 'camera', 'x1', 'y1', 'x2', 'y2']]
            bounding_boxes[bounding_boxes["obj_id"] != "0"].to_csv(os.path.join(OUTPUT_CSV_PATH, f"day_{day}_set_{set_id}.csv"), index=False)
            set_id += 1