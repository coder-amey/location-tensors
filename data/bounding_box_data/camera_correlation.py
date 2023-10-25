import pandas as pd

df = pd.DataFrame()
for day in range(1, 21):
    df = pd.concat([df, pd.read_csv(f"cross_camera_matches/day_{day}.csv")])

for cam in df.camera.unique().tolist():
    print(f"Camera {cam}: {df[df.camera == cam].shape}")
    print(df[df.camera == cam].next_cam.value_counts())
