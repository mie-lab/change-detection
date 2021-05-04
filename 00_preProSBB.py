import pandas as pd
import geopandas as gpd
import psycopg2
import numpy as np
from tqdm import tqdm
import os
import sys

from shapely.geometry import LineString
from shapely import wkt

from config import config

sys.path.append(os.path.join(os.getcwd(), "trackintel"))
from trackintel.preprocessing.triplegs import generate_trips


class DataReader:
    def __init__(self):
        self.conn = None
        self.trips = None
        self.tpls = None
        self.stps = None

    def read_from_database(self):
        # define the connection
        # fill in the credencials
        self.conn = psycopg2.connect(
            f"host='127.0.0.1' port='5433' dbname='commitdb' user='{config['commit_username']}' password='{config['commit_password']}'"
        )

        # read the trips
        print("Reading trips!")
        trips_df = pd.read_sql(
            "SELECT id, user_id AS userid, started_at AS startt, finished_at AS endt FROM gc1.trips", self.conn
        )

        # read the triplegs
        print("Reading triplegs!")
        sql_str = (
            "SELECT id, user_id AS userid, trip_id AS tripid, started_at AS startt, finished_at AS endt,"
            "mode_detected AS mdetect, mode_validated AS mvali, validated, geometry AS geom,"
            "ST_Length(geometry::geography) As length FROM gc1.triplegs WHERE ST_NumPoints(geometry) > 1"
        )
        triplegs_gdf = gpd.read_postgis(sql_str, self.conn)
        triplegs_gdf = triplegs_gdf.loc[triplegs_gdf.geometry.is_valid]
        triplegs_gdf = triplegs_gdf[triplegs_gdf["tripid"].notnull()]
        # datatype
        triplegs_gdf["tripid"] = triplegs_gdf["tripid"].astype("int64")
        triplegs_gdf["startt"] = pd.to_datetime(triplegs_gdf["startt"])
        triplegs_gdf["endt"] = pd.to_datetime(triplegs_gdf["endt"])
        triplegs_gdf["dur_s"] = (triplegs_gdf["endt"] - triplegs_gdf["startt"]).dt.total_seconds()
        # assign the correct mode to triplegs
        triplegs_gdf["mode"] = triplegs_gdf["mdetect"]
        idx = triplegs_gdf["validated"] == True
        triplegs_gdf.loc[idx, "mode"] = triplegs_gdf.loc[idx, "mvali"]
        triplegs_gdf.drop(columns=["mvali", "validated", "mdetect"], inplace=True)

        ## read the staypoints_df
        print("Reading staypoints!")
        sql_str = (
            "SELECT id, user_id AS userid, trip_id AS tripid, "
            "prev_trip_id AS ptripid, next_trip_id AS ntripid, "
            "started_at AS startt, finished_at AS endt, "
            "purpose_detected AS pdetect, purpose_validated AS pvali, "
            "validated, activity, geometry_raw AS geom FROM gc1.staypoints"
        )
        sp_df = gpd.read_postgis(sql_str, self.conn)
        sp_df = sp_df.loc[sp_df.geometry.is_valid]
        # select the activity staypoints
        sp_df.drop(index=sp_df[sp_df["activity"] != True].index, inplace=True)
        # time
        sp_df["startt"] = pd.to_datetime(sp_df["startt"])
        sp_df["endt"] = pd.to_datetime(sp_df["endt"])
        sp_df["dur_s"] = (sp_df["endt"] - sp_df["startt"]).dt.total_seconds()
        # delete negative staypoints
        sp_df.drop(index=sp_df[sp_df["dur_s"] < 0].index, inplace=True)
        # delete columns
        sp_df.drop(columns=["activity", "tripid"], inplace=True)
        # assign the correct purpose to staypoints
        sp_df["purp"] = sp_df["pdetect"]
        idx = sp_df["validated"] == True
        sp_df.loc[idx, "purp"] = sp_df.loc[idx, "pvali"]
        sp_df.drop(columns=["pvali", "validated", "pdetect"], inplace=True)

        self.conn.close()

        # save the tables
        trips_df.to_csv(os.path.join(config["S_raw"], "trip.csv"), index=False)
        triplegs_gdf.to_csv(os.path.join(config["S_raw"], "tpls.csv"), index=False)
        sp_df.to_csv(os.path.join(config["S_raw"], "stps.csv"), index=False)

        self.trips = trips_df
        self.tpls = triplegs_gdf
        self.stps = sp_df

    def matching(self):
        # raw records
        # self.trips = pd.read_csv(os.path.join(config["S_raw"], "trip.csv"))
        # self.tpls = pd.read_csv(os.path.join(config["S_raw"], "tpls.csv"))
        # self.stps = pd.read_csv(os.path.join(config["S_raw"], "stps.csv"))

        # regenerated records
        self.trips = pd.read_csv(os.path.join(config["S_raw2"], "trips.csv"))
        self.tpls = pd.read_csv(os.path.join(config["S_raw2"], "tpls.csv"))
        self.stps = pd.read_csv(os.path.join(config["S_raw2"], "stps.csv"))
        self.trips = self._from_trackintel(self.trips)
        self.tpls = self._from_trackintel(self.tpls)
        self.stps = self._from_trackintel(self.stps)
        self.tpls.rename(columns={"trip_id": "tripid"}, inplace=True)

        tqdm.pandas(desc="Load Geometry")
        self.tpls["geom"] = self.tpls["geom"].progress_apply(wkt.loads)
        merged_df = self._merge_triplegs().dropna(subset=["length_m"])

        # merge with trips_df to get the userid and time of trips
        merged_df = merged_df.merge(self.trips, on="id")

        merged_df.rename(columns={"origin_staypoint_id": "pstpid", "destination_staypoint_id": "nstpid"}, inplace=True)

        merged_df["startt"] = pd.to_datetime(merged_df["startt"])
        merged_df["endt"] = pd.to_datetime(merged_df["endt"])
        merged_df["dur_s"] = (merged_df["endt"] - merged_df["startt"]).dt.total_seconds()

        # save to file
        trip_for_Mode = merged_df.drop(columns="geometry")
        trip_for_Mode.to_csv(os.path.join(config["S_proc"], "trips_forMainMode.csv"), index=False)

        # final trip file
        merged_df = merged_df[
            ["geometry", "id", "userid", "startt", "endt", "pstpid", "nstpid", "dur_s", "length_m", "mode_d1"]
        ]
        # use distance as criteria for main mode
        merged_df.rename(columns={"mode_d1": "mode"}, inplace=True)
        merged_df.to_csv(os.path.join(config["S_proc"], "trips_wGeo.csv"), index=False)
        # without geo column
        merged_df.drop(columns="geometry").to_csv(os.path.join(config["S_proc"], "trips.csv"), index=False)

    def re_generate_trips(self):
        self.tpls = pd.read_csv(os.path.join(config["S_raw"], "tpls.csv"))
        self.stps = pd.read_csv(os.path.join(config["S_raw"], "stps.csv"))

        # filter out overlapping timelines
        self.filter_duplicates()

        # get the activity column, prepare for trip generation
        self.stps["activity"] = True
        self.stps.loc[(self.stps["purp"] == "wait"), "activity"] = False
        # self.stps.loc[(self.stps["purp"] == "unknown") | (self.stps["purp"] == "wait"), "activity"] = False
        self.stps.set_index("id", inplace=True)
        self.tpls.set_index("id", inplace=True)

        # run the trackintel trip generation
        stps, tpls, trips = generate_trips(self.stps, self.tpls, print_progress=True)

        stps.to_csv(os.path.join(config["S_raw2"], "stps.csv"), index=True)
        tpls.to_csv(os.path.join(config["S_raw2"], "tpls.csv"), index=True)
        trips.to_csv(os.path.join(config["S_raw2"], "trips.csv"), index=True)

    def filter_duplicates(self):
        self.tpls["startt"], self.tpls["endt"] = pd.to_datetime(self.tpls["startt"]), pd.to_datetime(self.tpls["endt"])
        self.stps["startt"], self.stps["endt"] = pd.to_datetime(self.stps["startt"]), pd.to_datetime(self.stps["endt"])
        self.tpls["dur_s"] = (self.tpls["endt"] - self.tpls["startt"]).dt.total_seconds()

        print(self.stps.columns)
        print(self.tpls.columns)
        # change to trackintel format
        stps = self._to_trackintel(self.stps)
        tpls = self._to_trackintel(self.tpls)
        print("User number:", len(stps["user_id"].unique()), len(tpls["user_id"].unique()))

        # merge trips and staypoints
        print("starting merge", stps.shape, tpls.shape)
        stps["type"] = "stp"
        tpls["type"] = "tpl"
        df_all = pd.merge(stps, tpls, how="outer")
        print("finished merge", df_all.shape)

        df_all = df_all.groupby("user_id", as_index=False).apply(self._alter_diff)
        self.stps = df_all.loc[df_all["type"] == "stp"].drop(columns=["type"])
        self.tpls = df_all.loc[df_all["type"] == "tpl"].drop(columns=["type"])
        self.stps = self.stps[["id", "user_id", "started_at", "finished_at", "geom", "duration", "purp"]]
        self.tpls = self.tpls[["id", "user_id", "started_at", "finished_at", "geom", "length", "duration", "mode"]]

    def _alter_diff(self, df):
        df.sort_values(by="started_at", inplace=True)
        df["diff"] = pd.NA
        df["st_next"] = pd.NA

        diff = df.iloc[1:]["started_at"].reset_index(drop=True) - df.iloc[:-1]["finished_at"].reset_index(drop=True)
        df["diff"][:-1] = diff.dt.total_seconds()
        df["st_next"][:-1] = df.iloc[1:]["started_at"].reset_index(drop=True)

        df.loc[df["diff"] < 0, "finished_at"] = df.loc[df["diff"] < 0, "st_next"]

        df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
        df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()

        # print(df.loc[df["diff"] < 0])
        df.drop(columns=["diff", "st_next"], inplace=True)
        df.drop(index=df[df["duration"] <= 0].index, inplace=True)

        return df

    def _to_trackintel(self, df):
        df.rename(
            columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at", "dur_s": "duration"},
            inplace=True,
        )
        # drop invalid
        df.drop(index=df[df["duration"] < 0].index, inplace=True)
        return df

    def _from_trackintel(self, df):
        df.rename(
            columns={"user_id": "userid", "started_at": "startt", "finished_at": "endt", "duration": "dur_s"},
            inplace=True,
        )
        return df

    # join list of coordinates to a string
    def _listToString(self, s):
        str1 = ","
        return str1.join(s)

    def _merge_linestrings(self, ls):
        res = []
        for line in ls:
            res.extend(line.coords)
        return LineString(res)

    def _get_mode(self, mode_ls):
        encode_dict = {
            "Mode::Airplane": 0,
            "Mode::Bicycle": 6,
            "Mode::Boat": 7,
            "Mode::Bus": 4,
            "Mode::Car": 5,
            "Mode::Coach": 2,
            "Mode::Ebicycle": 6,
            "Mode::Ecar": 5,
            "Mode::Ski": 7,
            "Mode::Train": 1,
            "Mode::Tram": 3,
            "Mode::Walk": 7,
        }
        decode_dict = {
            0: "Mode::Airplane",
            1: "Mode::Train",
            2: "Mode::Coach",
            3: "Mode::Tram",
            4: "Mode::Bus",
            5: "Mode::Car",
            6: "Mode::Bicycle",
            7: "Mode::Walk",
        }
        for mode, value in encode_dict.items():
            mode_ls = mode_ls.replace(mode, str(value))
        mode = np.min(list(map(int, mode_ls.split(","))))

        return decode_dict[mode]

    def _merge_triplegs_single(self, df):

        curr_id = df["id"].unique()[0]
        curr_df = self.tpls.loc[self.tpls["tripid"] == curr_id]

        res_dict = {}
        if not curr_df.empty:
            # sort the starting time
            sort_time = curr_df.sort_values(by="startt")
            # sort_dur = curr_df.sort_values(by="dur_s", ascending=False).head(2)
            sort_leng = curr_df.sort_values(by="length", ascending=False)

            mode_ls = self._listToString(sort_time["mode"].tolist())

            # res_dict['id'] = curr_id
            # length is the sum of triplegs -> no gap is included!
            res_dict["length_m"] = curr_df["length"].sum()
            # all modes, ranked according to time
            res_dict["mode_ls"] = mode_ls
            # main mode based on mode hierarchy
            res_dict["mode_hir"] = self._get_mode(mode_ls)

            # main mode based on time
            # res_dict["mode_t1"] = sort_dur["mode"].values[0]
            # res_dict["mode_tp1"] = sort_dur["dur_s"].values[0] / curr_df["dur_s"].sum()

            # main mode based on distance
            res_dict["mode_d1"] = sort_leng["mode"].values[0]
            res_dict["mode_dp1"] = sort_leng["length"].values[0] / curr_df["length"].sum()

            res_dict["mode_len"] = sort_leng["mode"].values
            res_dict["mode_lenProp"] = sort_leng["length"].values / curr_df["length"].sum()

            # link all points of triplegs together, might create gaps
            res_dict["geometry"] = self._merge_linestrings(sort_time["geom"].tolist())

            # store second used mode if neccessary
            # lens = sort_dur["mode"].values.shape[0]
            # if lens > 1:
            #     res_dict["mode_t2"] = sort_dur["mode"].values[1]
            #     res_dict["mode_tp2"] = sort_dur["dur_s"].values[1] / curr_df["dur_s"].sum()

            #     res_dict["mode_d2"] = sort_leng["mode"].values[1]
            #     res_dict["mode_dp2"] = sort_leng["length"].values[1] / curr_df["length"].sum()
            # else:
            #     res_dict["mode_t2"] = None
            #     res_dict["mode_tp2"] = None

            #     res_dict["mode_d2"] = None
            #     res_dict["mode_dp2"] = None

            # return trips -> id and triplegs -> trip_id
            return pd.Series(res_dict)
        # else:
        # trips that has no triplegs

    # merge triplegs and trip
    def _merge_triplegs(self):
        # related through trips -> id and triplegs -> trip_id
        tqdm.pandas(desc="trip generation")
        res = self.trips.groupby(["id"], as_index=False).progress_apply(self._merge_triplegs_single)
        return res


# generate a toy dataset with 10 users
def generateToyData(mtrips_gdf, points_gdf):
    # extract the first ten users
    user_10 = mtrips_gdf["user_id"].unique()[:10]

    # select the records from original dataset
    mtrips_gdf["flag"] = mtrips_gdf["user_id"].apply(lambda x: x in user_10)
    toy_10 = mtrips_gdf[mtrips_gdf["flag"]].drop(columns=["flag"])
    # save
    # toy_10.to_file("../data/dataset_toy/trips/trips.shp")

    points_gdf["flag"] = points_gdf["user_id"].apply(lambda x: x in user_10)
    points_10 = points_gdf[points_gdf["flag"]].drop(columns=["flag"])

    # points_10.to_file("../data/dataset_toy/points/points.shp")


if __name__ == "__main__":
    dataReader = DataReader()
    # dataReader.re_generate_trips()
    dataReader.matching()

