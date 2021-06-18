"""
Preprocessing for generating the data figure (Figure 2).
"""

import pandas as pd
import geopandas as gpd
import os, sys
from shapely import wkt
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "utils"))
from config import config

# load the tripleg file
tpls = pd.read_csv(os.path.join(config["raw"], "tpls.csv"))[["id", "geom", "mode"]]
print(tpls.head())

# change to geopandas geodataframe
tqdm.pandas(desc="Load Geometry")
tpls["geom"] = tpls["geom"].progress_apply(wkt.loads)
tpls = gpd.GeoDataFrame(tpls, geometry="geom")
tpls.set_crs("EPSG:4326", inplace=True)

print(len(tpls))

# exclude Ski and Airplane
tpls = tpls.loc[(tpls["mode"] != "Mode::Ski") & (tpls["mode"] != "Mode::Airplane")]

# change Boat to Bus
tpls.loc[(tpls["mode"] == "Mode::Boat"), "mode"] = "Mode::Bus"

# delete "Mode::"
tpls["mode"] = tpls["mode"].apply(lambda x: x[6:])
print(tpls["mode"].unique())

# save
tpls.to_file("tpls_mode.shp")

# reproject and save (caution: very slow!)
# extend = gpd.read_file(r"swiss_1903+.shp")
# tpls = tpls.to_crs(extend.crs)
# tpls.to_file(r"tpls_mode_proj.shp")

