import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
import datetime

from matplotlib import pyplot as plt
import matplotlib
from itertools import groupby

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13

from utils.config import config
from similarityMeasures import getValidTrips, _ifOnlyWalk

mode_dict = {
    "Mode::Airplane": "",
    "Mode::Bicycle": "Bike",
    "Mode::Boat": "Bus",
    "Mode::Bus": "Bus",
    "Mode::Car": "Car",
    "Mode::Coach": "Coach",
    "Mode::Ebicycle": "EBike",
    "Mode::Ecar": "ECar",
    "Mode::Ski": "",
    "Mode::Train": "Train",
    "Mode::Tram": "Tram",
    "Mode::Walk": "",
}


def classVisualization(user_df, window_size):
    """
    Visualize the Hierarchical clustering results.
    
    Including:
    1. mode frequency. 
    2. trip class count before and after filtering 
    3. distance and duration scatter plot of each class
    4. relative observation frequency evolution of each class
    """
    current_user = user_df["userid"].unique()[0]
    df, bestClusterNum = getHierarchicalResult(current_user)

    # create the analysis folder if not exist
    curr_path = config["resultFig"] + f"\\{current_user}"
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

    # organize modes
    ifOnlyWalk = df["mode_ls"].apply(_ifOnlyWalk)
    df.loc[ifOnlyWalk, "mode_encode"] = "Walk"
    df.loc[~ifOnlyWalk, "mode_encode"] = [_encodeMode(i, mode_dict) for i in df.loc[~ifOnlyWalk, "mode_ls"].to_list()]

    ## visualize trip counts per class
    # trip count per cluster before filtering
    df.groupby("cluster").size().plot.bar(figsize=(10, 5))
    plt.xlabel(f"#Cluster {bestClusterNum}")
    plt.ylabel("Trip count")
    plt.savefig(curr_path + "\\num.png", bbox_inches="tight")
    plt.close("all")

    df = filterClasses(df)

    # visualize trip count per cluster after filtering
    df.groupby("cluster").size().plot.bar(figsize=(10, 5))
    plt.xlabel(f"#Cluster {bestClusterNum}")
    plt.ylabel("Trip count")
    plt.savefig(curr_path + "\\num_filtered.png", bbox_inches="tight")
    plt.close("all")

    # visualize mode frequency plot per cluster
    mode_freq = df.groupby("cluster").apply(_getModeFrequency).dropna().reset_index()
    if len(mode_freq.columns) == 2:
        return None
    mode_freq.columns = ["clusters", "mode", "count"]
    _drawModeFreqency(mode_freq)
    plt.savefig(curr_path + "/freq.png", bbox_inches="tight")
    plt.close("all")

    # visualize 'length' and 'duration' scatter plot
    _drawScatter(df)
    plt.savefig(curr_path + "/stat.png", bbox_inches="tight", dpi=600)
    plt.close("all")

    # visualize the relative evolution of trip class usage
    classCountEvolution_df, _ = draw_relative(df, window_size)
    classCountEvolution_df.to_csv(curr_path + "/countEvolution.csv")
    plt.savefig(curr_path + "/count_rel.png", bbox_inches="tight", dpi=600)
    plt.close("all")


def getHierarchicalResult(user):
    """Get the result of the Hierarchical clustering: df contains trip set trips each with a class label."""
    cluster_path = config["cluster"] + f"\\{user}"

    # get the result of Hierarchical clustering
    filename = glob.glob(cluster_path + "\\H_N[0-9]*.csv")[0]
    best_num = int(filename.split("\\")[-1].split("_")[-1].split(".")[0][1:])
    df = pd.read_csv(filename)
    df = df[["id", "length_m", "mode_ls", "userid", "startt", "endt", f"hc_{best_num}"]]
    df.rename(columns={f"hc_{best_num}": "cluster"}, inplace=True)
    df["cluster"] = df["cluster"].astype("int")

    df["startt"] = pd.to_datetime(df["startt"])
    df["endt"] = pd.to_datetime(df["endt"])

    # time
    df["dur_min"] = (df["endt"] - df["startt"]).dt.total_seconds() / 60

    # distance
    df["length_km"] = df["length_m"] / 1000

    return df, best_num


def filterClasses(df):
    """Filter the final clustering result according to sample number and number of classes."""
    # include classes that contain more than 3% trips
    df = df.groupby("cluster").apply(_filter, len(df)).dropna().reset_index(drop=True)

    # only include top5 most observed classes
    if len(df["cluster"].unique()) > 5:
        top5 = df["cluster"].value_counts().head(5).index
        df = df.loc[df["cluster"].isin(top5)]
    df["cluster"] = df["cluster"].astype("int")
    return df


def draw_relative(df, window_size):
    """plot the relative distance and duration evolution of each package class."""

    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()

    count_dic = {"all": []}
    for key in np.sort(df["cluster"].unique()):
        count_dic.update({f"{key}": []})

    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # current trip
        c_df = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end)]

        # count
        count = c_df.groupby("cluster").size()
        for key, value in count_dic.items():
            if key == "all":
                value.append(count.sum())
                continue
            if int(key) in count.index.to_list():
                value.append(count.at[int(key)])
            else:
                value.append(0)

    idx = pd.date_range(
        start=start_date + datetime.timedelta(weeks=window_size), periods=weeks - window_size + 1, freq="W"
    )

    # count
    count_df = pd.DataFrame(count_dic, index=idx)
    # turn to relative
    relativeCount_df = count_df.div(count_df["all"] / 100, axis=0)
    relativeCount_df.drop(columns="all", inplace=True)
    relativeCount_df.plot(ylim=[0, 100])
    plt.ylabel("Trip proportion (%)", fontsize=16)
    # plt.legend("", frameon=False)
    plt.legend(loc="upper right", prop={"size": 13})
    # plt.xlabel('Time')

    return count_df, idx


def _encodeMode(ori_str, mode_dict):
    """Replace the ori_str to corresponding mode in mode_dict"""
    for mode, value in mode_dict.items():
        ori_str = ori_str.replace(mode, value)
    return ori_str.strip(",")


def _filter(df, trip_num):
    """Filter the final clustering result according to sample number"""
    # only include classes that contain more than 3% trips for visualization
    if df.shape[0] > trip_num * 0.03:
        return df


def _removeEmptyModes(modeSequence):
    """Remove empty modes ("") from the mode sequence."""
    result_str = []
    for (key, _) in groupby(modeSequence):
        result_str.append(key)
    return "".join(result_str)


def _getModeFrequency(df):
    """Get the average mode frequency existing in df."""
    df["mode_encode"] = df["mode_encode"].apply(_removeEmptyModes)
    return pd.DataFrame(df["mode_encode"].str.split(",").tolist(), index=df["id"]).stack().value_counts() / df.shape[0]


def _drawModeFreqency(df):
    """Plot the relative frequency of each mode."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    unique_mode = df["mode"].unique()
    clusters = df["clusters"].unique()
    _, axs = plt.subplots(1, clusters.shape[0], figsize=(20, 3), sharey=True)
    # plt.tight_layout()
    for i, ax in enumerate(axs):
        curr_clu = int(clusters[i])
        current_df = df.loc[df["clusters"] == curr_clu]
        bins = np.zeros_like(unique_mode)
        for j, mode in enumerate(unique_mode):
            count = current_df.loc[current_df["mode"] == mode]
            if not count.empty:
                bins[j] = count["count"].to_numpy()[0]
        ax.bar(unique_mode, bins, color=colors[i])
        ax.set_xlabel(f"Class {curr_clu}", fontsize=16)
        if i == 0:
            ax.set_ylabel("Average frequency", fontsize=16)


def _drawScatter(gdf):
    """plot the distance and duration scatter plot."""
    grouped = gdf.groupby("cluster")

    _, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, key in enumerate(grouped.groups.keys()):
        key = int(key)
        grouped.get_group(key)[["length_km", "dur_min"]].plot(
            kind="scatter", s=10, ax=ax, x="length_km", y="dur_min", c=colors[i], label=f"Class {key}"
        )
    plt.legend(loc="lower right", prop={"size": 13})
    plt.ylabel("Duration (min)", fontsize=16)
    plt.xlabel("Distance (km)", fontsize=16)
    plt.xlim([10 ** (-1.5), 10 ** (2.5)])
    plt.ylim([10 ** (-0.2), 10 ** (2.8)])

    plt.yscale("log")
    plt.xscale("log")


if __name__ == "__main__":
    time_window = 5

    t_df = getValidTrips(time_window=time_window)

    tqdm.pandas(desc="visualize behaviour classes")
    t_df.groupby("userid").progress_apply(classVisualization, window_size=time_window)

