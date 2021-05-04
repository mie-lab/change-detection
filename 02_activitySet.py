import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime
import sys, os, glob
import multiprocessing
from joblib import Parallel, delayed

from config import config

plt.rcParams["figure.dpi"] = 400


# get locations within the activity set
def _get_act_locs(df, time_window=20, filter_len=10):
    if df.shape[0] >= 2:
        avg_duration_min = df["dur_s"].sum() / 60 / time_window
        if avg_duration_min < filter_len:
            len_class = 0
        elif avg_duration_min < 30:
            len_class = 1
        elif avg_duration_min < 60:
            len_class = 2
        elif avg_duration_min < 60 * 6:
            len_class = 3
        elif avg_duration_min < 60 * 12:
            len_class = 4
        elif avg_duration_min < 60 * 24:
            len_class = 5
        elif avg_duration_min < 60 * 48:
            len_class = 6
        else:
            len_class = 7
        return pd.Series([avg_duration_min, len_class], index=["dur_s", "class"])


def get_curr_trips(t, stps, ASet):
    # get the locations in activity set
    valid_stps = stps.loc[stps["locid"].isin(ASet["locid"].unique())]

    # consider trip that ends in valid stps
    valid_t = t.loc[t["nstpid"].isin(valid_stps["id"])]
    valid_t = valid_t[["id", "length_m", "dur_s", "nstpid"]]

    # enrich with loc id
    valid_t = valid_t.merge(valid_stps[["id", "locid"]], left_on="nstpid", right_on="id")

    valid_t.drop(columns={"id_y", "nstpid"}, inplace=True)

    # enrich with activity set class
    valid_t = valid_t.merge(ASet[["locid", "class"]], on="locid", how="left")

    valid_t.rename(columns={"locid": "nloc", "id_x": "tripid"}, inplace=True)

    return valid_t


def applyParallel(dfGrouped, func, time_window, filter_len):
    # multiprocessing.cpu_count()
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group, time_window, filter_len) for name, group in tqdm(dfGrouped)
    )
    return pd.concat(retLst)


# get activity set for each user
def _extractActivitySet(time_window, filter_len):
    #
    stps_gdf = pd.read_csv(os.path.join(config["S_proc"], "stps_act_user_50.csv"))
    stps_gdf.rename(columns={"duration": "dur_s"}, inplace=True)

    # we need the trip mode sequence
    trips_gdf = pd.read_csv(os.path.join(config["S_proc"], "trips_forMainMode.csv"))

    # time
    trips_gdf["startt"], trips_gdf["endt"] = pd.to_datetime(trips_gdf["startt"]), pd.to_datetime(trips_gdf["endt"])
    stps_gdf["startt"], stps_gdf["endt"] = pd.to_datetime(stps_gdf["startt"]), pd.to_datetime(stps_gdf["endt"])

    trips_gdf["startt"] = pd.to_datetime(trips_gdf["startt"]).dt.tz_localize(None)
    trips_gdf["endt"] = pd.to_datetime(trips_gdf["endt"]).dt.tz_localize(None)
    stps_gdf["startt"] = pd.to_datetime(stps_gdf["startt"]).dt.tz_localize(None)
    stps_gdf["endt"] = pd.to_datetime(stps_gdf["endt"]).dt.tz_localize(None)

    trips_gdf["dur_s"] = (trips_gdf["endt"] - trips_gdf["startt"]).dt.total_seconds()

    # drop the columns
    stps_gdf.drop(columns={"activity", "trip_id"}, inplace=True)
    trips_gdf.drop(
        columns={
            "mode_hir",
            "mode_t1",
            "mode_tp1",
            "mode_d1",
            "mode_dp1",
            "mode_t2",
            "mode_tp2",
            "mode_d2",
            "mode_dp2",
        },
        inplace=True,
    )

    stps_gdf["type"] = "points"
    trips_gdf["type"] = "trips"
    all_ = trips_gdf.append(stps_gdf)

    # core
    tqdm.pandas(desc="Extracting activity set")
    allSet = applyParallel(all_.groupby("userid"), _extractActivitySetSingle, time_window, filter_len).reset_index(
        drop=True
    )
    tripStat = applyParallel(all_.groupby("userid"), _getTripStatSingle, time_window, filter_len).reset_index(drop=True)

    aSet = allSet.loc[allSet["type"] == "points"][["userid", "locid", "dur_s", "class", "timeStep"]]
    aSet.reset_index(drop=True, inplace=True)
    tSet = allSet.loc[allSet["type"] == "trips"][["userid", "tripid", "length_m", "dur_s", "nloc", "class", "timeStep"]]
    tSet.reset_index(drop=True, inplace=True)

    aSet.to_csv(os.path.join(config["S_act"], f"{time_window}_{filter_len}_aSet.csv"), index=False)
    tSet.to_csv(os.path.join(config["S_act"], f"{time_window}_{filter_len}_tSet.csv"), index=False)
    tripStat.to_csv(os.path.join(config["S_act"], f"{time_window}_{filter_len}_stat.csv"), index=False)


def _extractActivitySetSingle(df, time_window, filter_len):

    # total weeks and start week
    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()
    aSet = pd.DataFrame([], columns=["userid", "locid", "dur_s", "class", "timeStep"])
    tSet = pd.DataFrame([], columns=["userid", "tripid", "length_m", "dur_s", "nloc", "class", "timeStep"])

    # construct the sliding week gdf
    for i in range(0, weeks - time_window + 1):
        # start and end time
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=time_window), datetime.time())

        # currect gdf and extract the activity set
        curr_stps = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end) & (df["type"] == "points")]
        curr_ASet = (
            curr_stps.groupby("locid", as_index=False)
            .apply(_get_act_locs, time_window=time_window, filter_len=filter_len)
            .dropna()
        )

        if curr_ASet.empty:
            continue
        # result is the locations with stayed duration

        curr_ASet["timeStep"] = i
        aSet = aSet.append(curr_ASet)

        # for determine activity set trips
        curr_ASet = curr_ASet.loc[curr_ASet["class"] > 0]
        curr_t = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end) & (df["type"] == "trips")]
        curr_tSet = get_curr_trips(curr_t, curr_stps, curr_ASet)
        curr_tSet["timeStep"] = i
        tSet = tSet.append(curr_tSet)

    aSet.reset_index(drop=True)
    tSet.reset_index(drop=True)
    aSet["type"] = "points"
    tSet["type"] = "trips"
    aSet["userid"] = df["userid"].unique()[0]
    tSet["userid"] = df["userid"].unique()[0]
    return aSet.append(tSet)


def _getTripStatSingle(df, time_window, filter_len):
    # total weeks and start week
    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()

    res_ls = []
    # construct the sliding week gdf
    for i in range(0, weeks - time_window + 1):
        # start and end time
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=time_window), datetime.time())
        # currect gdf and extract the activity set
        curr_t = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end) & (df["type"] == "trips")]

        count = curr_t.shape[0]
        dis = curr_t["length_m"].sum()
        dur = curr_t["dur_s"].sum()
        res_ls.append([count, dis, dur, i])
    res = pd.DataFrame(res_ls, columns=["count", "dis", "dur", "timeStep"])
    res["userid"] = df["userid"].unique()[0]
    return res


def getSet():
    # MOBIS: 3, 4, 5, 6, 8
    # SBB: 4, 5, 6, 8, 10, 15, 20, 30, 40

    time_window_ls = [4, 5, 6, 8, 10, 15, 20, 30, 40]
    # 5, 10, 30
    filter_len_ls = [10]

    for time_window in time_window_ls:
        for filter_len in filter_len_ls:
            _extractActivitySet(time_window, filter_len)
            print(f"complete time_window{time_window} & filter_len{filter_len}")


if __name__ == "__main__":
    # get activity set
    getSet()

