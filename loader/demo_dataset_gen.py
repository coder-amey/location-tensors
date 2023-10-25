#External imports
from math import floor
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

#Internal imports
from utils.utils import store_pkl
from global_config.global_config import (
    ALL_BOUNDING_BOXES_PATH, CROSS_CAM_MATCHES_PATH, ENT_DEP_PATH, \
	TENSOR_DATA_PATH, NUM_DAYS, FEATURE_COLUMNS, T_MARGIN, MODE)

ORIGINAL_IMAGE_WIDTH = 1920
ORIGINAL_IMAGE_HEIGHT = 1200

PARTITION_WIDTH = ORIGINAL_IMAGE_WIDTH // 3
PARTITION_HEIGHT = ORIGINAL_IMAGE_HEIGHT // 3

def get_partition(x1, y1, x2, y2):
	x = (x1 + x2) / 2
	y = (y1 + y2) / 2
	part_num = str(int(floor(x / PARTITION_WIDTH))) + str(int(floor(y / PARTITION_HEIGHT)))
	return int(part_num, 3)

def partition_wrapper(row):
	return f"{row['camera']}_{get_partition(row['x1'], row['y1'], row['x2'], row['y2'])}"

def new_col_wrapper(cam_wise_df, field):
	new_field = cam_wise_df[field].value_counts()
	def inner_fn(value):
		return new_field.loc[value]
	return inner_fn


def get_id(day, hour, cam, track):
	return f"{day}_{hour}_{cam}_{track}"


def gen_id_wrapper(day, hour, cam):
	def inner_fn(track):
		return get_id(day, hour, cam, track)
	return inner_fn

if MODE == "Generate":
	print(f"Generating dataset...")
		
	# LOAD ALL OBJECTS AND FIND THEIR ID & PARTITION
	all_bbs = pd.DataFrame()
	sim_obj_ids = []
	for day in range(1, NUM_DAYS):
		print(f"Processing day: {day}")
		
		df1 = pd.read_csv(os.path.join(ALL_BOUNDING_BOXES_PATH, \
							f"all_bounding_boxes_day_{day}.csv")).astype(int)
		df2 = pd.read_csv(os.path.join(CROSS_CAM_MATCHES_PATH, \
							f"day_{day}.csv")).astype(int)
		hours = df1.hour.unique().tolist()
		hour_wise_groups = df1.groupby('hour')
		
		for hour, hour_wise_df in hour_wise_groups:
			cams = hour_wise_df.camera.unique().tolist()
			cam_wise_groups = hour_wise_df.groupby("camera")
			
			for cam, cam_wise_df in cam_wise_groups:
				# ASSIGN IDs
				cam_wise_df['day'] = day
				get_track_len = new_col_wrapper(cam_wise_df, "track")
				cam_wise_df["track_len"] = cam_wise_df.track.apply(get_track_len)
				gen_id = gen_id_wrapper(day, hour, cam)
				cam_wise_df["id"] = cam_wise_df.track.apply(gen_id)
				cam_wise_df = cam_wise_df[['id', 'day', 'hour', 'camera', 'track', 'frame_num', 'x1', 'y1', 'x2', 'y2', 'track_len']]
				all_bbs = pd.concat([all_bbs, cam_wise_df])

				# FILTER 1: Track length
				cam_wise_df = cam_wise_df.loc[cam_wise_df["track_len"] >= 24]
				
				# FILTER 2: Simultaneous objects
				get_frame_counts = new_col_wrapper(cam_wise_df, "frame_num")
				cam_wise_df["frame_counts"] = cam_wise_df["frame_num"].apply(get_frame_counts)
				sim_obj_ids += cam_wise_df.loc[cam_wise_df["frame_counts"] > 1, "id"].unique().tolist()

	# CLEAN THE CO-ORDINATES
	print(f"Post-ETL cleaning")
	all_bbs.x1 = all_bbs.x1.clip(lower=0, upper=ORIGINAL_IMAGE_WIDTH-1)
	all_bbs.x2 = all_bbs.x2.clip(lower=0, upper=ORIGINAL_IMAGE_WIDTH-1)
	all_bbs.y1 = all_bbs.y1.clip(lower=0, upper=ORIGINAL_IMAGE_HEIGHT-1)
	all_bbs.y2 = all_bbs.y2.clip(lower=0, upper=ORIGINAL_IMAGE_HEIGHT-1)
	all_bbs["partition"] = all_bbs[['camera', 'x1', 'y1', 'x2', 'y2']].apply(partition_wrapper, axis=1)

	# DEFINE DERIVED DATAFRAMES
	print(f"Derived DFs")
	all_objs = all_bbs[['id', 'day', 'hour', 'camera', 'track', 'track_len']].drop_duplicates().copy()
	all_objs = all_objs.set_index('id')

	df = all_bbs.sort_values(by=['id', 'frame_num'])\
		.drop_duplicates(subset='id', keep='first').set_index('id')
	all_objs["ent_frame"] = df["frame_num"].copy()
	all_objs["ent_cam_part"] = df["partition"].copy()

	df = all_bbs.sort_values(by=['id', 'frame_num'])\
		.drop_duplicates(subset='id', keep='last').set_index('id')
	all_objs["dep_frame"] = df["frame_num"].copy()
	all_objs["dep_cam_part"] = df["partition"].copy()

	sim_objs = all_objs[all_objs.index.isin(sim_obj_ids)].copy()

	# STORE THE DATAFRAMES
	all_bbs.to_csv("demo_dataset/all_boxes.csv", index=False)
	all_objs.to_csv("demo_dataset/all_objects.csv")
	sim_objs.to_csv("demo_dataset/simultaneous_objects.csv")

elif MODE == "Load":
	all_bbs = pd.read_csv("demo_dataset/all_boxes.csv")
	all_objs = pd.read_csv( "demo_dataset/all_objects.csv", index_col='id')
	sim_objs = pd.read_csv("demo_dataset/simultaneous_objects.csv", index_col='id')

print(f"All boxes:\n{all_bbs}\nAll objects:\n{all_objs}\nSimultaneous objects:\n{sim_objs}\n")

if MODE == "Generate":
	# LOAD THE CROSS-CAMERA MATCHES
	# Requires all_objs to be defined
	print(f"Matching objects...")
	matched_objs = pd.DataFrame( \
		columns=["dep_id", "dep_frame", "dep_cam_part", "ent_id", "ent_frame", "ent_cam_part"])
	for day in range(1, NUM_DAYS):
		print(f"Processing day: {day}")
		matches = pd.read_csv(os.path.join(CROSS_CAM_MATCHES_PATH, \
									f"day_{day}.csv")).astype(int)
		ents_deps = pd.read_csv(os.path.join(ENT_DEP_PATH, \
			f"entrances_and_departures_day_{day}.csv")).astype(int)
		for index, row in matches.iterrows():
			hour, cam, track = list(ents_deps.loc[row.departure_index] \
							[['hour', 'camera', 'track']].values)
			dep_id = get_id(day, hour, cam, track)
			try:
				dep_cam_part = all_objs.loc[dep_id, 'dep_cam_part']
				dep_frame = all_objs.loc[dep_id, 'dep_frame']
			except:
				print(f"Couldn't find exiting object: {dep_id}")
				continue
			
			hour, cam, track = list(ents_deps.loc[row.entrance_index, \
											['hour', 'camera', 'track']].values)
			ent_id = get_id(day, hour, cam, track)
			try:
				ent_cam_part = all_objs.loc[ent_id, 'ent_cam_part']
				ent_frame = all_objs.loc[ent_id, 'ent_frame']
			except:
				print(f"Couldn't find entering object: {ent_id}")
				continue
			
			matched_objs = matched_objs.append(pd.DataFrame([\
				[dep_id, dep_frame, dep_cam_part, ent_id, ent_frame, ent_cam_part]], \
					columns=matched_objs.columns), ignore_index=True)

	# INCLUDE THE TRANSIT TIME
	matched_objs["transit"] = matched_objs["ent_frame"] - matched_objs["dep_frame"]
	# Filter out hoax matches (negative transit times)
	matched_objs = matched_objs[matched_objs.transit > 0]
	
	# FIND POSTERIOR PROBABILITIES AND FILTER OUT THE LEAST SIGNIFICANT ONES
	conditional_probabilities = matched_objs.groupby('ent_cam_part')['dep_cam_part']\
		.value_counts().unstack().fillna(0).unstack()
	consolidated_ent_deps = conditional_probabilities[conditional_probabilities > 10]\
		.sort_values(ascending=False)
	print(consolidated_ent_deps)
	print(f"Total transitions: {consolidated_ent_deps.sum()}")

	# INCLUDE AVERAGE TRANSIT DELAYS
	average_transits = [matched_objs.loc[(matched_objs.dep_cam_part == dep_part) \
					& (matched_objs.ent_cam_part == ent_part), 'transit'].mean() \
					for dep_part, ent_part in consolidated_ent_deps.index.tolist()]
	consolidated_ent_deps = pd.DataFrame(data={'transit_counts': consolidated_ent_deps.tolist(), \
											'avg_transit': average_transits}, \
												index=consolidated_ent_deps.index)
	
	# STORE THE MATCH AND TRANSIT DATA
	matched_objs.to_csv("demo_dataset/matched_objects.csv", index=False)
	consolidated_ent_deps.to_csv("demo_dataset/consolidated_ent_deps.csv")

elif MODE == "Load":
	matched_objs = pd.read_csv("demo_dataset/matched_objects.csv")
	consolidated_ent_deps = pd.read_csv("demo_dataset/consolidated_ent_deps.csv", \
									 index_col=["dep_cam_part", "ent_cam_part"])

print("Matched objects:\n", matched_objs)
print("Consolidated entrances & departures:\n", consolidated_ent_deps)


if MODE == "Generate":
	# PERFORM FILTERING & CROSS-JOIN ON SIMULTANEOUS OBJECTS TO OBTAIN LIKELY TRANSITIONS
	## JOIN
	print("\nPerforming join...")
	sim_ent_dep_join = sim_objs.reset_index().merge(sim_objs.reset_index(), on=['day', 'hour'], \
													how='inner', \
													suffixes= ('_dep', '_ent'))
	sim_ent_dep_join = sim_ent_dep_join[sim_ent_dep_join.dep_frame_dep < sim_ent_dep_join.ent_frame_ent]
	sim_ent_dep_join["transit"] = sim_ent_dep_join.ent_frame_ent - sim_ent_dep_join.dep_frame_dep

	## FILTER
	print("Filtering matches...")
	likely_transits = pd.DataFrame(columns=sim_ent_dep_join.columns)
	for index, row in consolidated_ent_deps.iterrows():
		dep_part, ent_part = index
		avg_transit = row['avg_transit']
		df = sim_ent_dep_join[(sim_ent_dep_join.dep_cam_part_dep == dep_part) & \
				(sim_ent_dep_join.ent_cam_part_ent == ent_part) & \
				(sim_ent_dep_join.transit.between(avg_transit - T_MARGIN, avg_transit + T_MARGIN))]
		df["t_delta"] = abs(df.transit - avg_transit)
		likely_transits = pd.concat([likely_transits, df])

	likely_transits = likely_transits.sort_values(by="t_delta", ascending=True)

	# GENERATE THE DATASET OF CONFIRMED MATCHES
	confirmed_sim_matches = pd.DataFrame()
	## Iterate over the likely matches and pop the best fits
	total = likely_transits.shape[0]
	print("Finding best matches...")
	while not likely_transits.empty:
		print(f"\rProgress: {round(100*(1 - (likely_transits.shape[0])/total), 2)}%", end="")
		row = likely_transits.iloc[0]
		confirmed_sim_matches = confirmed_sim_matches.append(row, ignore_index=True)
		likely_transits = likely_transits[(likely_transits.id_dep != row.id_dep) & \
			(likely_transits.id_ent != row.id_ent)]
	print("\n")

	# OBTAIN ALL SIMULTANEOUS MATCHES AND FILTER OUT THE NON-SIMULTANEOUS ONES
	conf_sim_obj_ids = list(set(confirmed_sim_matches.id_dep.unique().tolist() + \
						confirmed_sim_matches.id_ent.unique().tolist()))
	conf_sim_bbs = all_bbs[all_bbs.id.isin(conf_sim_obj_ids)].copy()
	# FILTER: Simultaneous objects
	groups = conf_sim_bbs.groupby(['day', 'hour', 'camera'])
	sim_obj_ids = []
	for (day, hour, cam), cam_wise_df in groups:
		get_frame_counts = new_col_wrapper(cam_wise_df, "frame_num")
		cam_wise_df["frame_counts"] = cam_wise_df["frame_num"].apply(get_frame_counts)
		sim_obj_ids += cam_wise_df.loc[cam_wise_df["frame_counts"] > 1, "id"].unique().tolist()

	# Retain the simultaneous departures
	confirmed_sim_matches = confirmed_sim_matches[confirmed_sim_matches.id_dep.isin(sim_obj_ids)]
	# Re-derive the dataframes
	conf_sim_obj_ids = list(set(confirmed_sim_matches.id_dep.unique().tolist() + \
						confirmed_sim_matches.id_ent.unique().tolist()))
	conf_sim_bbs = all_bbs[all_bbs.id.isin(conf_sim_obj_ids)].copy()
	conf_sim_objs = all_objs[all_objs.index.isin(conf_sim_obj_ids)].copy()

	# STORE THE DATASET
	conf_sim_bbs.to_csv("demo_dataset/confirmed_simultaneous_boxes.csv", index=False)
	conf_sim_objs.to_csv("demo_dataset/confirmed_simultaneous_objects.csv")
	confirmed_sim_matches.to_csv("demo_dataset/confirmed_simultaneous_matches.csv", index=False)

elif MODE == "Load":
	conf_sim_bbs = pd.read_csv("demo_dataset/confirmed_simultaneous_boxes.csv")
	conf_sim_objs = pd.read_csv("demo_dataset/confirmed_simultaneous_objects.csv", index_col='id')
	confirmed_sim_matches = pd.read_csv("demo_dataset/confirmed_simultaneous_matches.csv")

print(f"Simultaneous boxes:\n{conf_sim_bbs}\nSimultaneous objects:\n{conf_sim_objs}\nSimultaneous matches:\n{confirmed_sim_matches}\n")

# GENERATE THE TRAJECTORIES
if MODE == "Generate":
	demo_trajectories = {id: conf_sim_bbs.loc[conf_sim_bbs.id == id, \
							FEATURE_COLUMNS].to_numpy() \
							for id in conf_sim_objs.index.tolist()}
	store_pkl(demo_trajectories, os.path.join(TENSOR_DATA_PATH, \
										   "demo_trajectories.pkl"))
