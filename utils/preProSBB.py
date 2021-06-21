import pandas as pd
import os
import datetime
from tqdm import tqdm

from shapely.geometry import LineString
from shapely import wkt

from config import config


def match_trips():
    """Match trips and triplegs with id."""

    # read records
    trips = pd.read_csv(os.path.join(config["raw"], "trips.csv"))
    tpls = pd.read_csv(os.path.join(config["raw"], "tpls.csv"))

    # load Geometry
    tqdm.pandas(desc="Load Geometry")
    tpls["geom"] = tpls["geom"].progress_apply(wkt.loads)

    # relate trips with triplegs
    tqdm.pandas(desc="trip generation")
    merged_df = tpls.groupby(["tripid"], as_index=False).progress_apply(_merge_triplegs)

    # merge with trips_df to get the userid and time of trips
    merged_df = merged_df.join(trips.set_index("id"))

    merged_df.rename(
        columns={"origin_staypoint_id": "pstpid", "destination_staypoint_id": "nstpid", "tripid": "id"}, inplace=True
    )
    # time
    merged_df["startt"] = pd.to_datetime(merged_df["startt"])
    merged_df["endt"] = pd.to_datetime(merged_df["endt"])
    merged_df["dur_s"] = (merged_df["endt"] - merged_df["startt"]).dt.total_seconds()

    # end period cut
    end_period = datetime.datetime(2017, 12, 25)
    merged_df = merged_df.loc[merged_df["endt"] < end_period]

    # save trip
    if not os.path.exists(config["proc"]):
        os.makedirs(config["proc"])
    merged_df.drop(columns="geometry").to_csv(os.path.join(config["proc"], "trips.csv"), index=False)
    merged_df.to_csv(os.path.join(config["proc"], "trips_wGeo.csv"), index=False)


def _merge_triplegs(df):

    res_dict = {}
    # sort the starting time
    sort_time = df.sort_values(by="startt")

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
    match_trips()
