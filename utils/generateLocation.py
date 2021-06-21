import pandas as pd
import datetime
import geopandas as gpd
import os, sys
from tqdm import tqdm
from shapely import wkt


from config import config

sys.path.append(os.path.join(os.getcwd(), "trackintel"))
import trackintel as ti


def generate_Location(df, epsilon, user):
    """Cluster staypoints to locations, with different parameters and distinguish 'user' and 'dataset'"""

    # select only activity staypoints
    print(df.shape)
    df = df.loc[df["activity"] == True].copy()
    print(df.shape)

    # change to trackintel format
    df.set_index("id", inplace=True)
    tqdm.pandas(desc="Load Geometry")
    df["geom"] = df["geom"].progress_apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry="geom")

    # cluster the staypoints into locations (DBSCAN)
    if user:
        agg_level = "user"
    else:
        agg_level = "dataset"

    stps, locs = gdf.as_staypoints.generate_locations(
        epsilon=epsilon, num_samples=1, distance_metric="haversine", agg_level=agg_level
    )
    print("cluster complete")
    # rename to avoid conflict
    stps.rename(
        columns={"user_id": "userid", "started_at": "startt", "finished_at": "endt", "location_id": "locid"},
        inplace=True,
    )

    locs.rename(columns={"user_id": "userid"}, inplace=True)

    stps["startt"] = pd.to_datetime(stps["startt"]).dt.tz_localize(None)
    stps["endt"] = pd.to_datetime(stps["endt"]).dt.tz_localize(None)

    stps.sort_index(inplace=True)
    locs.sort_index(inplace=True)

    stps.to_csv(os.path.join(config["proc"], f"stps_act_{agg_level}_{epsilon}.csv"), index=True)
    locs.to_csv(os.path.join(config["proc"], f"locs_{agg_level}_{epsilon}.csv"), index=True)


if __name__ == "__main__":
    # SBB
    # df = pd.read_csv(os.path.join(config["raw"], "stps.csv"))
    # df.rename(columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at"}, inplace=True)
    # df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
    # # end period cut
    # end_period = datetime.datetime(2017, 12, 25)
    # df = df.loc[df["finished_at"] < end_period].copy()
    # df["started_at"] = df["started_at"].dt.tz_localize(tz="utc")
    # df["finished_at"] = df["finished_at"].dt.tz_localize(tz="utc")

    # Geolife
    df = pd.read_csv(os.path.join(config["proc"], "stps.csv"))
    df.rename(columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at"}, inplace=True)
    df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])

    generate_Location(df, epsilon=50, user=True)
