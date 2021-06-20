import pandas as pd
from tqdm import tqdm
import datetime
import os
import multiprocessing
from joblib import Parallel, delayed

from utils.config import config


def getSets(stps_gdf, trips_gdf, time_window_ls):
    """Get activity and trip sets for different time_window_ls."""
    stps_gdf["type"] = "points"
    trips_gdf["type"] = "trips"
    all_ = trips_gdf.append(stps_gdf)

    for time_window in time_window_ls:
        allSet = extractSetsSingle(all_, time_window)
        print(f"complete time_window {time_window}")

        # clean up
        aSet = allSet.loc[allSet["type"] == "points"][["userid", "locid", "dur_s", "class", "timeStep"]]
        aSet.reset_index(drop=True, inplace=True)
        tSet = allSet.loc[allSet["type"] == "trips"][
            ["userid", "tripid", "length_m", "dur_s", "nloc", "class", "timeStep"]
        ]
        tSet.reset_index(drop=True, inplace=True)

        # save
        aSet.to_csv(os.path.join(config["activitySet"], f"{time_window}_aSet.csv"), index=False)
        tSet.to_csv(os.path.join(config["activitySet"], f"{time_window}_tSet.csv"), index=False)


def _getActLocs(df, time_window=20, filter_len=10):
    """Definition of locations within the activity set."""
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


def _getCurrTrips(t, stps, ASet):
    """Get the current trips that has travelled to an activity set location."""
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


def applyParallel(dfGrouped, func, time_window):
    """Parallel version of the groupby function"""
    # multiprocessing.cpu_count()
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group, time_window) for _, group in dfGrouped)
    return pd.concat(retLst)


def extractSetsSingle(all_, time_window):
    """Get activity set for a given time_window"""
    tqdm.pandas(desc="Extracting activity set")
    allSet = applyParallel(all_.groupby("userid"), _extractSetsSingleUser, time_window).reset_index(drop=True)

    return allSet


def _extractSetsSingleUser(df, time_window):
    """Get activity set and trip set for each individual."""

    # total weeks and start week
    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()
    aSet = pd.DataFrame([], columns=["userid", "locid", "dur_s", "class", "timeStep"])
    tSet = pd.DataFrame([], columns=["userid", "tripid", "length_m", "dur_s", "nloc", "class", "timeStep"])

    # construct the sliding week gdf, i is the timestep
    for i in range(0, weeks - time_window + 1):
        # start and end time
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=time_window), datetime.time())

        ## determine activity set locations
        # get the currect time step points gdf
        curr_stps = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end) & (df["type"] == "points")]
        # extract the activity set (location)
        curr_ASet = curr_stps.groupby("locid", as_index=False).apply(_getActLocs, time_window=time_window).dropna()

        # if no location, jump to next time step
        if curr_ASet.empty:
            continue

        # result is the locations with stayed duration class
        curr_ASet["timeStep"] = i
        aSet = aSet.append(curr_ASet)

        ## determine activity set trips
        # select activity set location
        curr_ASet = curr_ASet.loc[curr_ASet["class"] > 0]

        # get the currect time step trips gdf
        curr_t = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end) & (df["type"] == "trips")]
        curr_tSet = _getCurrTrips(curr_t, curr_stps, curr_ASet)

        # result is the trips that ends at activity set locations
        curr_tSet["timeStep"] = i
        tSet = tSet.append(curr_tSet)

    # clean up
    aSet.reset_index(drop=True)
    tSet.reset_index(drop=True)
    aSet["type"] = "points"
    tSet["type"] = "trips"
    aSet["userid"] = df["userid"].unique()[0]
    tSet["userid"] = df["userid"].unique()[0]
    return aSet.append(tSet)


if __name__ == "__main__":
    stps_gdf = pd.read_csv(os.path.join(config["proc"], "stps_act_user_50.csv"))
    trips_gdf = pd.read_csv(os.path.join(config["proc"], "trips.csv"))

    # time
    trips_gdf["startt"], trips_gdf["endt"] = pd.to_datetime(trips_gdf["startt"]), pd.to_datetime(trips_gdf["endt"])
    stps_gdf["startt"], stps_gdf["endt"] = pd.to_datetime(stps_gdf["startt"]), pd.to_datetime(stps_gdf["endt"])

    time_window_ls = [5, 10, 15]
    getSets(stps_gdf, trips_gdf, time_window_ls)
