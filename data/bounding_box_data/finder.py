import pandas as pd

print(f"day\thour\tcamera\ttrack\tframe\tTracked")
for day in range(1, 2):
    df1 = pd.read_csv(f"bounding_boxes/all_bounding_boxes_day_{day}.csv")
    df2 = pd.read_csv(f"cross_camera_matches/day_{day}.csv")
    hours = df1.hour.unique().tolist()
    partitions = df1.groupby('hour')
    partitions = [partitions.get_group(x).drop(['hour'], axis=1) for x in partitions.groups]
    cam_track = []

    for df1, hour in zip(partitions, hours):
        u_frames = df1.frame_num.unique().tolist()
        for frame in u_frames:
            if df1[df1.frame_num == frame].size > 1:
                concurrent_df1 = df1.loc[df1.frame_num == frame, ["camera", "track"]]
                cams_tracks = list(concurrent_df1.itertuples(index=False, name=None))
                for ct_tuple in cams_tracks:
                    if ct_tuple not in cam_track:
                            cam_track.append(ct_tuple)
                            if not df2.loc[(df2["hour"] == hour) & (df2["camera"] == ct_tuple[0]) & (df2["track"] == ct_tuple[1])].empty:
                                print(f"{day}\t{hour}\t{ct_tuple[0]}\t{ct_tuple[1]}\t{frame}\tTrue")
                            else:
                                 print(f"{day}\t{hour}\t{ct_tuple[0]}\t{ct_tuple[1]}\t{frame}\tFalse")
                            