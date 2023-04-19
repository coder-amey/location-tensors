from loader.loader import load_dataset
from loader.ETL import bbs2trajectories

# obj_trajectories = bbs2trajectories()
dataset = load_dataset(save_to_file=False)