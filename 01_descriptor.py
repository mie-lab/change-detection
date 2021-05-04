import skmob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from shapely import wkt

import sys, os

sys.path.append(os.path.join(os.getcwd(), "trackintel"))
from trackintel.preprocessing.staypoints import generate_locations

from config import config


# get the coordinate for places
def _calculate_center_coordinate(df):
    # the coord of points is the mean of staypoints in that cluster
    df["clat"] = df.groupby("cluster")["lat"].transform("mean")
    df["clng"] = df.groupby("cluster")["lng"].transform("mean")
    return df


# cluster staypoints to locations, with different parameters and distinguish 'user' and 'dataset'
def clusterLocation():
    all = False
    epsilon = 50

    # for MOBIS
    df = pd.read_csv(os.path.join(config["M_raw"], "stpsTrip.csv"))
    df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
    # for SBB
    # df = pd.read_csv(os.path.join(config["S_raw2"], "stps.csv"))
    # df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
    # df["started_at"] = df["started_at"].dt.tz_localize(tz="utc")
    # df["finished_at"] = df["finished_at"].dt.tz_localize(tz="utc")

    df = df.loc[df["activity"] == True]
    df.set_index("id", inplace=True)
    tqdm.pandas(desc="Bar")
    df["geom"] = df["geom"].progress_apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry="geom")
    # cluster the staypoints into locations (DBSCAN)
    if all:
        stps, locs = gdf.as_staypoints.generate_locations(
            epsilon=epsilon, num_samples=1, distance_matrix_metric="haversine", agg_level="dataset"
        )
    else:
        stps, locs = gdf.as_staypoints.generate_locations(
            epsilon=epsilon, num_samples=1, distance_matrix_metric="haversine", agg_level="user"
        )
    print(np.sort(stps["location_id"].unique())[:10])
    print("cluster complete")
    # rename to avoid conflict
    stps.rename(
        columns={"user_id": "userid", "started_at": "startt", "finished_at": "endt", "location_id": "locid"},
        inplace=True,
    )
    locs.rename(columns={"user_id": "userid"}, inplace=True)

    stps.sort_index(inplace=True)
    locs.sort_index(inplace=True)

    if all:
        stps.to_csv(os.path.join(config["S_proc"], f"stps_act_{epsilon}.csv"), index=True)
        locs.to_csv(os.path.join(config["S_proc"], f"locs_{epsilon}.csv"), index=True)
    else:
        stps.to_csv(os.path.join(config["M_proc"], f"stps_act_user_{epsilon}.csv"), index=True)
        locs.to_csv(os.path.join(config["M_proc"], f"locs_user_{epsilon}.csv"), index=True)


if __name__ == "__main__":

    # get locations from staypoints
    clusterLocation()

    # the descriptors that can be calculated with skmob library
    # skmobIndicator()

    # basic statatistics
    # temporalIndicators()

