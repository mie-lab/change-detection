import os
import pandas as pd
import pickle
import multiprocessing
from tqdm import tqdm

from utils.config import config
from getActivitySet import getSets
from similarityMeasures import getValidTrips, similarityMeasurement
from clustering import cluster
from clusterVisualization import classVisualization
from changeDetection import changeDetection

### define parameter

# time_window for activity set generation
time_window_ls = [5]

# the chosen time_window for similarity measurement, clustering and analysis
time_window = 5

# weights for similarity measurement
mode_weight = 1 / 3
distance_weight = 1 / 3
duration_weight = 1 / 3

# weights for change detection
slidingThres = 0.3  # 30%
lag = 5  # window
threshold = 3
influence = 1

if __name__ == "__main__":

    ### Step 1: extract activity sets
    stps_gdf = pd.read_csv(os.path.join(config["proc"], "stps_act_user_50.csv"))
    trips_gdf = pd.read_csv(os.path.join(config["proc"], "trips.csv"))

    trips_gdf["startt"], trips_gdf["endt"] = pd.to_datetime(trips_gdf["startt"]), pd.to_datetime(trips_gdf["endt"])
    stps_gdf["startt"], stps_gdf["endt"] = pd.to_datetime(stps_gdf["startt"]), pd.to_datetime(stps_gdf["endt"])
    stps_gdf["dur_s"] = (stps_gdf["endt"] - stps_gdf["startt"]).dt.total_seconds()
    getSets(stps_gdf, trips_gdf, time_window_ls)
    print("Finished activity set extraction.")

    ### Step 2: measure pairwise similarity of trips
    t_df = getValidTrips(time_window=time_window, SBB=False)

    all_dict = similarityMeasurement(
        t_df, mode_weight=mode_weight, distance_weight=distance_weight, duration_weight=duration_weight
    )

    # save the combined distance matrix
    if not os.path.exists(config["similarity"]):
        os.makedirs(config["similarity"])
    with open(config["similarity"] + f"/similarity.pkl", "wb") as f:
        pickle.dump(all_dict, f, pickle.HIGHEST_PROTOCOL)
    print("Finished similarity measurements.")

    ### Step 3: clustering with the similarity matrix
    t_df = getValidTrips(time_window=time_window, SBB=False)

    ## open distance matrix
    with open(config["similarity"] + f"/similarity.pkl", "rb") as f:
        dist_mat = pickle.load(f)

    ## parallel clustering
    jobs = []
    for user in t_df["userid"].unique():

        # get the distance matrix
        dist = dist_mat[user]["all"]

        # get the trip ID
        ut_df = t_df.loc[t_df["userid"] == user]

        # create the user folder
        curr_path = config["cluster"] + f"\\{user}"
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)

        # perform clustering
        p = multiprocessing.Process(target=cluster, args=(dist, ut_df, curr_path))

        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print("Finished clustering.")

    ### Step 4 (Optional): ploting of clustering results
    t_df = getValidTrips(time_window=time_window, SBB=False)

    tqdm.pandas(desc="visualize behaviour classes")
    t_df.groupby("userid").progress_apply(classVisualization, window_size=time_window)
    print("Finished cluster visualization.")

    ### Step 5: change detection based on the cluster evolution
    t_df = getValidTrips(time_window=time_window, SBB=False)

    tqdm.pandas(desc="change detection")
    detectionResults = t_df.groupby("userid").progress_apply(
        changeDetection,
        window_size=time_window,
        slidingThres=slidingThres,
        lag=lag,
        threshold=threshold,
        influence=influence,
    )
    detectionResults.to_csv(config["resultFig"] + "\\detectionResults.csv")
    print(detectionResults)
    print("Finished change detection.")
