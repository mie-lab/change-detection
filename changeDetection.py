import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import datetime


from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

from utils.config import config
from similarityMeasures import getValidTrips
from clusterVisualization import getHierarchicalResult, filterClasses, draw_relative


def changeDetection(user_df, window_size, slidingThres, lag, threshold, influence):
    """
    Change detection for each individual.
    
    Including sliding window based change detection and HHI index based change detection.
    """
    current_user = user_df["userid"].unique()[0]

    #
    tripClass_df, _ = getHierarchicalResult(current_user)
    # create the analysis folder if not exist
    curr_path = config["resultFig"] + f"\\{current_user}"
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

    # filter classes, here >3% and not larger than 5
    tripClass_df = filterClasses(tripClass_df)

    ### sliding window change detection
    changePeriods = _slidingWindowDetection(tripClass_df, window_size=window_size, threshold=slidingThres)
    # plot
    _, idx = draw_relative(tripClass_df, window_size)
    for change_point in changePeriods.to_dict("records"):
        plt.axvspan(xmin=idx[change_point["start"]], xmax=idx[change_point["end"]], color="green", alpha=0.2)
    plt.legend("", frameon=False)
    plt.savefig(curr_path + "/count_rel_change.png", bbox_inches="tight", dpi=600)
    plt.close()

    ### HHI change detection
    peaks, HHI_ls, idx = _HHIDetection(
        tripClass_df, window_size=window_size, lag=lag, threshold=threshold, influence=influence
    )
    _plotHHIDetection(peaks, HHI_ls, idx, curr_path)

    return pd.Series([changePeriods.values, peaks["signals"]], index=["windowDetectionPeriods", "HHIDetectionSignals"])


def _HHIDetection(df, window_size=5, lag=5, threshold=3, influence=1):
    """Calculate HHI index of df for each window_size, and detect anomaly signals using __thresholdingAlgo."""
    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()

    HHI_ls = []
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # current trip
        c_df = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end)]

        HHI_ls.append(__getHHI(c_df))

    peaks = __thresholdingAlgo(HHI_ls, lag=lag, threshold=threshold, influence=influence)
    idx = pd.date_range(
        start=start_date + datetime.timedelta(weeks=window_size), periods=weeks - window_size + 1, freq="W"
    )

    return peaks, HHI_ls, idx


def _plotHHIDetection(peaks, HHI_ls, idx, curr_path):
    """
    Plot the result of the HHI change dection.
    
    Including:
    1. HHI index, moving mean and upper/lower bound time evolution
    2. Detected signals.
    """

    HHI_df = pd.DataFrame(HHI_ls, columns=["HHI"], index=idx)

    peaks["avgFilter"] = np.insert(peaks["avgFilter"], 0, 0)
    peaks["stdFilter"] = np.insert(peaks["stdFilter"], 0, 0)
    HHI_df["avgFilter"] = peaks["avgFilter"][:-1]
    HHI_df["stdFilter"] = peaks["stdFilter"][:-1]
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

    signal_df = pd.DataFrame(peaks["signals"], columns=["signal"], index=idx)
    signal_df["signal"].plot(color="red", figsize=(6.4, 2))
    plt.ylabel("Signal", fontsize=16)
    plt.yticks([-1, 0, 1])
    plt.savefig(curr_path + "/signals.png", bbox_inches="tight", dpi=600)
    plt.close()


def _slidingWindowDetection(df, window_size, threshold):
    """Detect the minimum change period where any class has changed larger than threshold."""

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


def __getHHI(df):
    """HHI index calculation."""
    prop = df["cluster"].value_counts(normalize=True).values
    return np.sum(prop ** 2)


def __thresholdingAlgo(y, lag, threshold, influence):
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


if __name__ == "__main__":

    time_window = 5
    slidingThres = 0.3
    lag = 5
    threshold = 3
    influence = 1

    t_df = getValidTrips(time_window=time_window)

    tqdm.pandas(desc="change detection")
    detectionResults = t_df.groupby("userid").progress_apply(
        changeDetection,
        window_size=time_window,
        slidingThres=slidingThres,
        lag=lag,
        threshold=threshold,
        influence=influence,
    )
    print(detectionResults)
