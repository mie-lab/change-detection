import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import os
import glob
import datetime


from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13

from utils.config import config
from similarityMeasures import getValidTrips
from clusterAnalysis import getHierarchicalResult


def HHI_index(df):
    prop = df["cluster"].value_counts(normalize=True).values
    return np.sum(prop ** 2)


def thresholding_algo(y, lag, threshold, influence):
    """
    Peak detection algorithm.
    From https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data
    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1) : i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1) : i + 1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1) : i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1) : i + 1])

    return dict(signals=np.asarray(signals), avgFilter=np.asarray(avgFilter), stdFilter=np.asarray(stdFilter))


def _HHI(df):

    window_size = 5

    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()

    HHI_ls = []
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # current trip
        c_df = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end)]

        HHI_ls.append(HHI_index(c_df))

    peaks = thresholding_algo(HHI_ls, lag=5, threshold=3, influence=1)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    idx = pd.date_range(
        start=start_date + datetime.timedelta(weeks=window_size), periods=weeks - window_size + 1, freq="W"
    )
    HHI_df = pd.DataFrame(HHI_ls, columns=["HHI"], index=idx)

    HHI_df["avgFilter"] = 0
    HHI_df["stdFilter"] = 0
    HHI_df["avgFilter"].iloc[1:] = peaks["avgFilter"][:-1]
    HHI_df["stdFilter"].iloc[1:] = peaks["stdFilter"][:-1]
    # HHI_df["avgFilter"] = peaks["avgFilter"]
    # HHI_df["stdFilter"] = peaks["stdFilter"]
    HHI_df["HHI"].plot(label="HHI", color=colors[0])

    HHI_df["avgFilter"].plot(label="Moving mean", figsize=(6.4, 2), color=colors[1], alpha=0.5)

    (HHI_df["avgFilter"] + 3 * HHI_df["stdFilter"]).plot(label="Upper/lower bound", color="grey", alpha=0.5)
    (HHI_df["avgFilter"] - 3 * HHI_df["stdFilter"]).plot(color="grey", alpha=0.5)
    plt.legend(["HHI", "Moving mean", "upper/lower bound"])

    plt.ylabel("HHI index", fontsize=16)
    plt.ylim([0.1, 0.6])
    # plt.show()
    plt.savefig(curr_path + "/HHI.png", bbox_inches="tight", dpi=600)
    plt.close()

    # plt.figure(figsize=(6.4, 2.4))

    signal_df = pd.DataFrame(peaks["signals"], columns=["signal"], index=idx)
    signal_df["signal"].plot(color="red", figsize=(6.4, 2))
    plt.ylabel("Signal", fontsize=16)
    plt.yticks([-1, 0, 1])
    plt.savefig(curr_path + "/signals.png", bbox_inches="tight", dpi=600)
    plt.close()

    ## for ploting timestep where the top1 location changed
    # for i, change_point in enumerate(home_change_points):
    #     if change_point == True:
    #         plt.vlines(x=idx[i + 1], ymin=0, ymax=1, color="red", alpha=0.8)

    return HHI_ls


def _sliding_window(df, threshold):

    window_size = 5

    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()

    dist_ls = []
    # construct the sliding week gdf, to get the dist_ls (distribution for each timestep)
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # current trip
        c_df = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end)]
        # print(c_df)

        cluster_num = c_df.groupby("cluster").size().to_frame("Size")
        distribution = cluster_num / cluster_num.sum()
        dist_ls.append(distribution)

    change_ls = []

    # start to ensure no overlapping change time
    start = 0
    curr_max = 0
    hold_start = -1
    hold_change = -1
    find_subset = False
    for i, curr_dist in enumerate(dist_ls):

        if not find_subset:
            # find the starting j
            curr_max = 0
            change_pre = 0
            for j in range(i - 1, start - 1, -1):
                combined = curr_dist.join(dist_ls[j], lsuffix="l", rsuffix="r")
                change = np.abs(combined["Sizel"] - combined["Sizer"])

                # add small term to allow pertubation
                if change.max() + 0.05 < change_pre:
                    break
                change_pre = change.max()
                if change.max() > threshold:
                    if change.max() > curr_max:
                        hold_start = j
                        curr_max = change.max()
                        hold_change = change.max()
                        find_subset = True
                    else:
                        break

                # cut stable periods and ensure no super long change periods
                if (change < 0.05).all() and (i - j) > 15:
                    start = j - 1
                    break
        else:
            # find the ending i
            combined = curr_dist.join(dist_ls[hold_start], lsuffix="l", rsuffix="r")
            change = np.abs(combined["Sizel"] - combined["Sizer"])
            # print(change.max())
            if change.max() > hold_change:
                hold_change = change.max()
                if i < len(dist_ls) - 1:
                    continue

            if i == len(dist_ls) - 1:
                end = i
            else:
                end = i - 1
            change_ls.append([hold_start, end])
            start = end
            hold_change = -1
            find_subset = False

    change_points = pd.DataFrame(change_ls, columns=["start", "end"])
    return change_points


def _filter(df, trip_num):
    """filter the final clustering result according to sample number"""
    if df.shape[0] > trip_num * 0.03:
        return df


def _draw_relative_with_change(df, change_points):
    """Plot the relative count evolution of each package"""
    window_size = 5

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
            if int(float(key)) in count.index.to_list():
                value.append(count.at[int(float(key))])
            else:
                value.append(0)

    idx = pd.date_range(
        start=start_date + datetime.timedelta(weeks=window_size), periods=weeks - window_size + 1, freq="W"
    )

    change_points_dict = change_points.to_dict("records")

    # count
    count_df = pd.DataFrame(count_dic, index=idx)
    count_df.to_csv(curr_path + "/count_rel.csv")
    # turn to relative
    count_df = count_df.div(count_df["all"] / 100, axis=0)
    count_df.drop(columns="all", inplace=True)

    ## plot the relative evolution
    count_df.plot(ylim=[0, 100])
    for change_point in change_points_dict:
        plt.axvspan(xmin=idx[change_point["start"]], xmax=idx[change_point["end"]], color="green", alpha=0.2)

    plt.ylabel("Trip proportion (%)", fontsize=16)
    plt.legend("", frameon=False)
    # plt.legend(loc="upper right", prop={"size": 13})
    # plt.xlabel("Time")
    plt.savefig(curr_path + "/count_rel_change.png", bbox_inches="tight", dpi=600)
    plt.close()

    ## for getting the time and length of the first change period
    # first = change_points.sort_values(by="start").head(1)
    # if len(first) > 0:
    #     if first["start"].values[0] < 5:
    #         return (first["end"] - first["start"]).values[0]
    # return 0

    ## for ploting timestep where the top1 location changed
    # for i, change_point in enumerate(home_change_points):
    #     if change_point == True:
    #         plt.vlines(x=idx[i + 1], ymin=0, ymax=100, color="red", alpha=0.8)


# define which case to consider
case = "case3"

t_df = getValidTrips(time_window=5)

res = []

users = t_df["userid"].unique()
for user in tqdm(users):

    df, bestClusterNum = getHierarchicalResult(user)

    # create the analysis folder
    curr_path = config["S_fig"] + f"\\{case}_5\\" + str(user)
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

    # filter classes, here >3% and not larger than 5
    df = df.groupby("cluster").apply(_filter, len(df)).dropna().reset_index(drop=True)
    if len(df["cluster"].unique()) > 5:
        top5 = df["cluster"].value_counts().head(5).index
        df = df.loc[df["cluster"].isin(top5)]

    # sliding window change detection
    change_points = _sliding_window(df, threshold=0.30)
    # _draw_relative_with_change(df, change_points, home_change_points)
    _draw_relative_with_change(df, change_points)
    # ranges = _draw_relative_with_change(df, change_points)
    # if ranges != 0:
    #     res.append(ranges)

    # HHI change detection
    # HHI_ls = get_HHI_sequence(df, home_change_points)
    HHI_ls = _HHI(df)

## claculate change period statistics
# res = np.array(res)
# print(res.mean(), res.min(), res.max())
# print(len(res))
# plt.hist(res, bins=20)
# plt.show()
# print(np.median(res))
# print(np.std(res))
