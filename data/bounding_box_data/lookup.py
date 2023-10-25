import pandas as pd

print(f"day\thour\tcamera\ttrack\tframe\tTracked")
for day in range(1, 21):
    print(f"Day: {day}")
    df1 = pd.read_csv(f"bounding_boxes/all_bounding_boxes_day_{day}.csv").astype(int)
    df2 = pd.read_csv(f"cross_camera_matches/day_{day}.csv").astype(int)
    hours = df1.hour.unique().tolist()
    partitions = df1.groupby('hour')
    partitions = [partitions.get_group(x).drop(['hour'], axis=1) for x in partitions.groups]
    obj_count = {}

    for df1, hour in zip(partitions, hours):
        multi_obj_frames = df1["frame_num"].value_counts()
        multi_obj_frames = multi_obj_frames[multi_obj_frames > 1]
        obj_count[hour] = {}
        
        multi_obj_frames = df1.loc[df1["frame_num"].isin(multi_obj_frames.index), ["camera", "frame_num"]]
        # Partition on cameras
        cam_wise_groups = multi_obj_frames.groupby("camera")
        cam_wise_groups = [cam_wise_groups.get_group(x) for x in cam_wise_groups.groups]
        
        for cam_wise_df in cam_wise_groups:
            cam_wise_dict = cam_wise_df.value_counts().to_dict()
            cam_wise_dict = {k: v for k, v in \
                    cam_wise_dict.items() if v > 1}
            obj_count[hour].update(cam_wise_dict)
            
        
        """
        print(f"Day: 1\thour: {hour}")
        print(f"Cam\tframe\tcount")
        for tup, count in obj_count[hour].items():
            print(tup[0], tup[1], count, sep="\t")
        """
    print(f"Hour\tCam\tframe\tcount")
    for index, row in df2.iterrows():
        hour = row["hour"]
        cam = row["camera"]
        frame = row["frame_num"]
        if (cam, frame) in obj_count[hour].keys():
            print(hour, cam, frame, obj_count[hour][(cam, frame)], sep="\t")
    