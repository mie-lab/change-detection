import os
import pandas as pd
from tqdm import tqdm
from shapely.geometry import LineString

import trackintel as ti
from trackintel.io.dataset_reader import read_geolife, geolife_add_modes_to_triplegs
from trackintel.geogr.distances import calculate_haversine_length


from config import config


def readGeolife():
    """Read geolife dataset, generate trips and merge with tripleg information."""
    pfs, mode_labels = read_geolife("./geolife/Data", print_progress=True)
    # get user with mode labels
    user_with_mode = []
    for key, value in mode_labels.items():
        if len(value) > 0:
            user_with_mode.append(key)
    # only select user who has mode labels
    pfs = pfs.loc[pfs["user_id"].isin(user_with_mode)].copy()

    # generate staypoints, triplegs and trips
    pfs, spts = pfs.as_positionfixes.generate_staypoints(time_threshold=5.0, print_progress=True)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts)
    tpls = geolife_add_modes_to_triplegs(tpls, mode_labels)

    spts = ti.analysis.labelling.create_activity_flag(spts, time_threshold=15)

    spts, tpls, trips = ti.preprocessing.triplegs.generate_trips(spts, tpls, gap_threshold=15)
    spts.rename(columns={"user_id": "userid", "started_at": "startt", "finished_at": "endt"}, inplace=True)
    if not os.path.exists(config["proc"]):
        os.makedirs(config["proc"])
    spts.to_csv(os.path.join(config["proc"], "stps.csv"))

    # get the length of each tripleg
    tpls["length_m"] = calculate_haversine_length(tpls)

    # relate trips with triplegs
    tpls["mode"].fillna("", inplace=True)
    tqdm.pandas(desc="trips and triplegs merging")
    merged_df = tpls.groupby(["trip_id"], as_index=False).progress_apply(_merge_triplegs)

    # merge with trips_df to get the userid and time of trips
    merged_df = merged_df.join(trips)

    merged_df.rename(
        columns={
            "origin_staypoint_id": "pstpid",
            "destination_staypoint_id": "nstpid",
            "trip_id": "id",
            "user_id": "userid",
            "started_at": "startt",
            "finished_at": "endt",
        },
        inplace=True,
    )
    # filter out trips with no mode
    merged_df = merged_df[merged_df["mode"] != ""]

    # time
    merged_df["startt"] = pd.to_datetime(merged_df["startt"]).dt.tz_localize(None)
    merged_df["endt"] = pd.to_datetime(merged_df["endt"]).dt.tz_localize(None)
    merged_df["dur_s"] = (merged_df["endt"] - merged_df["startt"]).dt.total_seconds()

    # save trip
    merged_df.drop(columns="geometry").to_csv(os.path.join(config["proc"], "trips.csv"), index=False)
    merged_df.to_csv(os.path.join(config["proc"], "trips_wGeo.csv"), index=False)


def _merge_triplegs(df):

    res_dict = {}
    # sort the starting time
    sort_time = df.sort_values(by="started_at")

    # sort by length
    sort_leng = df.sort_values(by="length_m", ascending=False)

    mode_ls = ",".join(sort_time["mode"].tolist())

    # length is the sum of triplegs -> no temporal gap is included!
    res_dict["length_m"] = df["length_m"].sum()

    # all modes, ranked according to starting time
    res_dict["mode_ls"] = mode_ls

    # main mode based on distance
    res_dict["mode"] = sort_leng["mode"].values[0]
    res_dict["mode_p"] = sort_leng["length_m"].values[0] / df["length_m"].sum()

    # mode number
    res_dict["mode_len"] = len(sort_leng["mode"].values)

    # link all points of triplegs together, might create gaps
    res_dict["geometry"] = _merge_linestrings(sort_time["geom"].tolist())

    # return trips -> id and triplegs -> trip_id
    return pd.Series(res_dict)


def _merge_linestrings(ls):
    res = []
    for line in ls:
        res.extend(line.coords)
    return LineString(res)


if __name__ == "__main__":
    readGeolife()
