import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm

import pickle
import multiprocessing
import os

from scipy.spatial.distance import pdist, squareform
import itertools

from utils.config import config

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# replace and change into a single string
def _encode(ori_str, mode_dict):
    for mode, value in mode_dict.items():
        ori_str = ori_str.replace(mode, value)
    # join adjacent same mode
    return "".join(i for i, _ in itertools.groupby(ori_str.replace(",", "")))


def _getModeDist(df, return_dict, distance="jd"):
    """
    distance matrix calculation.
    distance = ['symm', "jd"]
    """

    n = len(df)

    # for efficiency, calculate only upper triangle matrix

    ix_1, ix_2 = np.triu_indices(n, k=1)
    trilix = np.tril_indices(n, k=-1)

    # initialize
    d = []
    D = np.zeros((n, n))

    ix_1_this = -1
    for i in tqdm(range(len(ix_1))):
        if ix_1[i] != ix_1_this:
            ix_1_this = ix_1[i]
            # traj_1 = np.asarray(gdf.iloc[ix_1_this].geometry)
            mode_1 = df.iloc[ix_1_this]["mode"]

        ix_2_this = ix_2[i]
        # traj_2 = np.asarray(gdf.iloc[ix_2_this].geometry)
        mode_2 = df.iloc[ix_2_this]["mode"]
        if distance == "symm":
            set_1 = set(mode_1)
            set_2 = set(mode_2)
            if len(set_1) + len(set_2) == 0:
                d.append(0)
            else:
                d.append(len(set_1.symmetric_difference(set_2)) / (len(set_1) + len(set_2)))
        if distance == "jd":
            set_1 = set(mode_1)
            set_2 = set(mode_2)
            if len(set_1) + len(set_2) == 0:
                d.append(0)
            else:
                dist = 1 - len(set_1.intersection(set_2)) / len(set_1.union(set_2))
                d.append(dist)

    d = np.asarray(d)
    D[(ix_1, ix_2)] = d

    # mirror triangle matrix to be conform with scikit-learn format
    D[trilix] = D.T[trilix]

    return_dict[df["userid"].unique()[0]] = D


def _getLengthDist(df):
    return squareform(pdist(df["length_m"].to_numpy().reshape(-1, 1), metric="euclidean"))


def _getDurDist(df):
    return squareform(pdist(df["dur_s"].to_numpy().reshape(-1, 1), metric="euclidean"))


def _ifOnlyWalk(row):
    """return True if only one walk mode is used."""
    modes = set(row.split(","))
    walkExist = "Mode::Walk" in modes
    length = len(modes) == 1
    return walkExist & length


def similarityMeasurement(t_df, mode_weight=0.5, distance_weight=0.25, duration_weight=0.25):

    ## define how to select modes
    # every mode, Boat = Bus, no airplane + no walk mode if combined with other mode
    mode_dict = {
        "Mode::Airplane": "",
        "Mode::Bicycle": "b",
        "Mode::Boat": "d",
        "Mode::Bus": "d",
        "Mode::Car": "e",
        "Mode::Coach": "f",
        "Mode::Ebicycle": "g",
        "Mode::Ecar": "h",
        "Mode::Ski": "",
        "Mode::Train": "i",
        "Mode::Tram": "j",
        "Mode::Walk": "",
    }

    ifOnlyWalk = t_df["mode_ls"].apply(_ifOnlyWalk)
    # only walk is assigned k
    t_df.loc[ifOnlyWalk, "mode"] = "k"
    # other trips ignore walk
    t_df.loc[~ifOnlyWalk, "mode"] = [_encode(i, mode_dict) for i in t_df.loc[~ifOnlyWalk, "mode_ls"].to_list()]

    ## similarity for mode
    manager = multiprocessing.Manager()
    mode_dict = manager.dict()
    jobs = []

    for user in t_df["userid"].unique():
        user_df = t_df.loc[t_df["userid"] == user]
        p = multiprocessing.Process(target=_getModeDist, args=(user_df, mode_dict, "jd"))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print("Mode distance number:", len(mode_dict))

    ## similarity for length
    len_dict = t_df.groupby("userid").apply(_getLengthDist).to_dict()

    ## similarity for duration
    dur_dict = t_df.groupby("userid").apply(_getDurDist).to_dict()

    ## combined similarity
    all_dict = {}
    # k is the user, v the mode pairwise distance matrix
    for k, v in mode_dict.items():
        all_dict[k] = {}

        # min-max for length
        length = (len_dict[k] - len_dict[k].min()) / (len_dict[k].max() - len_dict[k].min())
        all_dict[k]["len"] = length

        # min-max for duration
        duration = (dur_dict[k] - dur_dict[k].min()) / (dur_dict[k].max() - dur_dict[k].min())
        all_dict[k]["dur"] = duration

        # min-max for mode
        v = (v - v.min()) / (v.max() - v.min())
        all_dict[k]["mode"] = v

        # weight 2:1:1
        all_dict[k]["all"] = (
            all_dict[k]["mode"] * mode_weight
            + all_dict[k]["len"] * distance_weight
            + all_dict[k]["dur"] * duration_weight
        )

    return all_dict


if __name__ == "__main__":
    time_window = 5
    actTrips_df = pd.read_csv(os.path.join(config["activitySet"], f"{time_window}_tSet.csv"))
    trips_df = pd.read_csv(
        os.path.join(config["proc"], "trips.csv"),
        usecols=["id", "length_m", "mode_ls", "userid", "startt", "endt", "dur_s"],
    )
    # time
    trips_df["startt"], trips_df["endt"] = pd.to_datetime(trips_df["startt"]), pd.to_datetime(trips_df["endt"])

    # remove duplicate entries
    actTrips = actTrips_df["tripid"].unique()
    # t_df is the trip dataframe for similarity measures
    t_df = trips_df.loc[trips_df["id"].isin(actTrips)].copy()

    ## select only valid users
    valid_user = pd.read_csv(config["quality"] + "\\SBB_user_window_filtered.csv")["user_id"].unique()
    valid_user = valid_user.astype(int)
    t_df = t_df.loc[t_df["userid"].isin(valid_user)]
    print("User number:", t_df["userid"].unique().shape[0])

    # Core
    all_dict = similarityMeasurement(t_df)

    # save the combined distance matrix
    with open(config["similarity"] + f"/similarity.pkl", "wb") as f:
        pickle.dump(all_dict, f, pickle.HIGHEST_PROTOCOL)

