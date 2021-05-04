# preprocessing for generating the data figure


import pandas as pd
import geopandas as gpd
import os
import sys
from shapely import wkt
from tqdm import tqdm


from config import config

sys.path.append(os.path.join(os.getcwd(), "trackintel"))
import trackintel as ti


tpls = pd.read_csv(os.path.join(config["S_raw2"], "tpls.csv"))[["id", "geom", "mode"]]

tqdm.pandas(desc="Load Geometry")
tpls["geom"] = tpls["geom"].progress_apply(wkt.loads)
tpls = gpd.GeoDataFrame(tpls, geometry="geom")
tpls.set_crs("EPSG:4326", inplace=True)

print(len(tpls))

tpls = tpls.loc[(tpls["mode"] != "Mode::Ski") & (tpls["mode"] != "Mode::Airplane")]
tpls.loc[(tpls["mode"] == "Mode::Boat"), "mode"] = "Mode::Bus"
tpls["mode"] = tpls["mode"].apply(lambda x: x[6:])
print(tpls["mode"].unique())

extend = gpd.read_file(r"swiss_1903+.shp")
tpls = tpls.to_crs(extend.crs)
tpls.to_file(r"tpls_mode_proj.shp")

