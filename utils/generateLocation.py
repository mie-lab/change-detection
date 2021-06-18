import pandas as pd
import datetime
import geopandas as gpd
import os, sys
from tqdm import tqdm
from shapely import wkt


from config import config


sys.path.append(os.path.join(os.getcwd(), "trackintel"))
import trackintel as ti


def generate_Location(epsilon, user):
    """Cluster staypoints to locations, with different parameters and distinguish 'user' and 'dataset'"""

    df = pd.read_csv(os.path.join(config["raw"], "stps.csv"))
    df.rename(columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at"}, inplace=True)
    df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])

    # end period cut
    end_period = datetime.datetime(2017, 12, 25)
    df = df.loc[df["finished_at"] < end_period]

    df["started_at"] = df["started_at"].dt.tz_localize(tz="utc")
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz="utc")

    # select only activity staypoints
    df = df.loc[df["activity"] == True]

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
    generate_Location(epsilon=50, user=True)
