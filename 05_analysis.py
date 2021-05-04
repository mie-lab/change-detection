import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import os
import glob
import datetime

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from itertools import groupby


from config import config

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13
# matplotlib.rcParams.update({'figure.autolayout': True})

# get best cluster number according to WB_index
def _get_optimal_cluster_num(filename):
    df = pd.read_csv(filename)
    return df.loc[df["WB_index"] == df["WB_index"].min(), "number"].to_numpy()[0]


# replace the ori_str to corresponding mode in mode_dict
def _encode(ori_str, mode_dict):
    for mode, value in mode_dict.items():
        ori_str = ori_str.replace(mode, value)
    # return ''.join(i for i, _ in itertools.groupby(ori_str))
    return ori_str.strip(",")


# filter the final clustering result according to sample number
def _filter(df, trip_num):
    if df.shape[0] > trip_num * 0.03:
        # if df.shape[0] > 3:
        return df


def _remove_all_consecutive(str1):
    result_str = []
    for (key, group) in groupby(str1):
        result_str.append(key)

    return "".join(result_str)


# get the splitted mode df from original df
def _mode(df):
    df["mode_encode"] = df["mode_encode"].apply(_remove_all_consecutive)
    return pd.DataFrame(df["mode_encode"].str.split(",").tolist(), index=df["id"]).stack().value_counts() / df.shape[0]


# enrich the original gdf with temporal and spatial info
def _enrich(gdf):
    gdf = gdf.to_crs({"init": "epsg:2056"})
    gdf["length"] = gdf["geometry"].apply(lambda x: x.length) / 1000

    gdf["started_at"] = pd.to_datetime(gdf["started_at"])
    gdf["finished_a"] = pd.to_datetime(gdf["finished_a"])
    gdf["duration"] = (gdf["finished_a"] - gdf["started_at"]) / np.timedelta64(1, "h")

    gdf["year"] = pd.DatetimeIndex(gdf["started_at"]).year
    gdf["month"] = pd.DatetimeIndex(gdf["started_at"]).month

    return gdf


# plot the relative frequency of each mode
def _draw_freq(df):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    unique_mode = df["mode"].unique()
    clusters = df["clusters"].unique()
    _, axs = plt.subplots(1, clusters.shape[0], figsize=(20, 3), sharey=True)
    # plt.tight_layout()
    for i, ax in enumerate(axs):
        curr_clu = int(clusters[i])
        current_df = df.loc[df["clusters"] == curr_clu]
        bins = np.zeros_like(unique_mode)
        for j, mode in enumerate(unique_mode):
            count = current_df.loc[current_df["mode"] == mode]
            if not count.empty:
                bins[j] = count["count"].to_numpy()[0]
        ax.bar(unique_mode, bins, color=colors[i])
        ax.set_xlabel(f"Class {curr_clu}", fontsize=16)
        if i == 0:
            ax.set_ylabel("Average frequency", fontsize=16)


# plot the distance and duration scatter plot
def _draw_stat(gdf):
    grouped = gdf.groupby("cluster")
    ncols = 2
    nrows = int(np.ceil(grouped.ngroups / ncols))

    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,8), sharey=True, sharex=True)
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # max_len = gdf['length'].max()
    # min_len = gdf['length'].min()
    # bins = np.arange(min_len,max_len,(max_len-min_len)/30)
    # for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
    #     grouped.get_group(key)[['length','duration']].plot(kind='scatter', ax=ax, x='length', y='duration', label=f'Class {key}')
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')
    # fig.text(0.5, 0.04, 'Length (km)', ha='center')
    # fig.text(0.04, 0.5, 'Duration (h)', va='center', rotation='vertical')
    for i, key in enumerate(grouped.groups.keys()):
        key = int(key)
        grouped.get_group(key)[["length_km", "dur_min"]].plot(
            kind="scatter", s=10, ax=ax, x="length_km", y="dur_min", c=colors[i], label=f"Class {key}"
        )
    plt.legend(loc="lower right", prop={"size": 13})
    plt.ylabel("Duration (min)", fontsize=16)
    plt.xlabel("Distance (km)", fontsize=16)
    plt.xlim([10 ** (-1.5), 10 ** (2.5)])
    plt.ylim([10 ** (-0.2), 10 ** (2.8)])

    plt.yscale("log")
    plt.xscale("log")


# plot the absolute distance and duration evolution
def _draw_absolute(df, all_df):
    window_size = 5
    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()

    dis_dic = {"Total": [], "Considered": []}
    dur_dic = {"Total": [], "Considered": []}
    count_dic = {"Total": [], "Considered": []}
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # all trip
        cAll_df = all_df.loc[(all_df["startt"] >= curr_start) & (all_df["endt"] < curr_end)]
        # current trip
        c_df = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end)]

        # count
        count_dic["Total"].append(len(cAll_df) / window_size)
        count_dic["Considered"].append(len(c_df) / window_size)

        # duration
        dur_dic["Total"].append(cAll_df["dur_min"].sum() / window_size)
        dur_dic["Considered"].append(c_df["dur_min"].sum() / window_size)

        # distance
        dis_dic["Total"].append(cAll_df["length_km"].sum() / window_size)
        dis_dic["Considered"].append(c_df["length_km"].sum() / window_size)

    idx = pd.date_range(
        start=start_date + datetime.timedelta(weeks=window_size), periods=weeks - window_size + 1, freq="W"
    )

    count_df = pd.DataFrame(count_dic, index=idx)
    count_df.plot()
    count_df.to_csv(curr_path + "/count_abs.csv")
    plt.ylabel("Count (Trip/week)", fontsize=16)
    plt.xlabel("")
    plt.ylim([0, plt.ylim()[1] + 7])
    plt.legend(loc="upper right", prop={"size": 13})
    # plt.show()
    plt.savefig(curr_path + "/count_abs.png", bbox_inches="tight")
    plt.close()

    # dur_df = pd.DataFrame(dur_dic, index=idx)
    # dur_df.plot()
    # dur_df.to_csv(curr_path + "/dur_abs.csv")
    # plt.ylabel("Duration (min/week)", fontsize=16)
    # plt.xlabel("")
    # plt.ylim([0, plt.ylim()[1] + 7])
    # plt.legend(loc="upper right", prop={"size": 13})
    # # plt.show()
    # plt.savefig(curr_path + "/dur_abs.png", bbox_inches="tight")
    # plt.close()

    # dis_df = pd.DataFrame(dis_dic, index=idx)
    # dis_df.plot()
    # dis_df.to_csv(curr_path + "/dis_abs.csv")
    # plt.ylabel("Distance (km/week)", fontsize=16)
    # plt.xlabel("")
    # plt.ylim([0, plt.ylim()[1] + 300])
    # plt.legend(loc="upper right", prop={"size": 13})
    # # plt.show()
    # plt.savefig(curr_path + "/dis_abs.png", bbox_inches="tight")
    # plt.close()


# plot the relative distance and duration evolution of each package
def _draw_relative(df):
    window_size = 5

    weeks = (df["endt"].max() - df["startt"].min()).days // 7
    start_date = df["startt"].min().date()

    dur_dic = {"all": []}
    dis_dic = {"all": []}
    count_dic = {"all": []}
    for key in np.sort(df["cluster"].unique()):
        dur_dic.update({f"{key}": []})
        dis_dic.update({f"{key}": []})
        count_dic.update({f"{key}": []})
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # current trip
        c_df = df.loc[(df["startt"] >= curr_start) & (df["endt"] < curr_end)]

        # count
        count = c_df.groupby("cluster").size()
        for key, value in count_dic.items():
            if key == "all":
                value.append(count.sum())
                continue
            if int(key) in count.index.to_list():
                value.append(count.at[int(key)])
            else:
                value.append(0)

        # # duration
        # dur = c_df.groupby("cluster")["dur_min"].sum()
        # for key, value in dur_dic.items():
        #     if key == "all":
        #         value.append(c_df["dur_min"].sum())
        #         continue
        #     if int(key) in dur.index.to_list():
        #         value.append(dur.at[int(key)])
        #     else:
        #         value.append(0)

        # # distance
        # dis = c_df.groupby("cluster")["length_km"].sum()
        # for key, value in dis_dic.items():
        #     if key == "all":
        #         value.append(c_df["length_km"].sum())
        #         continue
        #     if int(key) in dis.index.to_list():
        #         value.append(dis.at[int(key)])
        #     else:
        #         value.append(0)

    idx = pd.date_range(
        start=start_date + datetime.timedelta(weeks=window_size), periods=weeks - window_size + 1, freq="W"
    )

    # count
    count_df = pd.DataFrame(count_dic, index=idx)
    count_df.to_csv(curr_path + "/count_rel.csv")
    # turn to relative
    count_df = count_df.div(count_df["all"] / 100, axis=0)
    count_df.drop(columns="all", inplace=True)
    count_df.plot(ylim=[0, 100])
    plt.ylabel("Trip proportion (%)", fontsize=16)
    # plt.legend("", frameon=False)
    plt.legend(loc="upper right", prop={"size": 13})
    # plt.xlabel('Time')
    plt.savefig(curr_path + "/count_rel.png", bbox_inches="tight", dpi=600)
    plt.close()

    # # duration
    # dur_df = pd.DataFrame(dur_dic, index=idx)
    # dur_df.to_csv(curr_path + "/dur_rel.csv")
    # # turn to relative
    # dur_df = dur_df.div(dur_df["all"] / 100, axis=0)
    # dur_df.drop(columns="all", inplace=True)
    # dur_df.plot(ylim=[0, 100])
    # plt.ylabel("Duration (%)", fontsize=16)
    # plt.legend(loc="upper right", prop={"size": 13})
    # # plt.xlabel('Time')
    # plt.savefig(curr_path + "/dur_rel.png", bbox_inches="tight", dpi=600)
    # plt.close()

    # # distance
    # dis_df = pd.DataFrame(dis_dic, index=idx)
    # dis_df.to_csv(curr_path + "/dis_rel.csv")

    # dis_df = dis_df.div(dis_df["all"] / 100, axis=0)
    # dis_df.drop(columns="all", inplace=True)
    # dis_df.plot(ylim=[0, 100])
    # plt.ylabel("Distance (%)", fontsize=16)
    # plt.legend(loc="upper right", prop={"size": 13})
    # # plt.xlabel('Time')
    # plt.savefig(curr_path + "/dis_rel.png", bbox_inches="tight", dpi=600)
    # plt.close()


def _ifOnlyWalk(row):
    modes = set(row.split(","))
    walkExist = "Mode::Walk" in modes
    length = len(modes) == 1
    return walkExist & length


mode_dict = {
    "Mode::Airplane": "",
    "Mode::Bicycle": "Bike",
    "Mode::Boat": "Bus",
    "Mode::Bus": "Bus",
    "Mode::Car": "Car",
    "Mode::Coach": "Coach",
    "Mode::Ebicycle": "EBike",
    "Mode::Ecar": "ECar",
    "Mode::Ski": "",
    "Mode::Train": "Train",
    "Mode::Tram": "Tram",
    "Mode::Walk": "",
}

# define which case to consider
case = "case3"
actTrips_df = pd.read_csv(os.path.join(config["S_act"], "5_10_tSet.csv"))
trips_df = pd.read_csv(
    os.path.join(config["S_proc"], "trips_forMainMode.csv"),
    usecols=["id", "length_m", "mode_ls", "userid", "startt", "endt"],
)
# remove duplicate entries
actTrips = actTrips_df["tripid"].unique()
t_df = trips_df.loc[trips_df["id"].isin(actTrips)].copy()

# valid users
valid_user = pd.read_csv(config["results"] + "\\SBB_user_window_filtered.csv")["user_id"].unique()
valid_user = valid_user.astype(int)
t_df = t_df.loc[t_df["userid"].isin(valid_user)]

# use only 20 user first
users = t_df["userid"].unique()

# all trips: trips_df
trips_df["startt"] = pd.to_datetime(trips_df["startt"])
trips_df["endt"] = pd.to_datetime(trips_df["endt"])
trips_df["dur_min"] = (trips_df["endt"] - trips_df["startt"]).dt.total_seconds() / 60
trips_df["length_km"] = trips_df["length_m"] / 1000

for user in tqdm(users):
    curr_path = config["S_cluster"] + f"\\{case}_cluster_5\\" + str(user)

    # get the result of hierarchy clustering
    filename = glob.glob(curr_path + "\\H_N[0-9]*.csv")[0]
    best_num = int(filename.split("\\")[-1].split("_")[-1].split(".")[0][1:])
    df = pd.read_csv(filename)
    df = df[["id", "length_m", "mode_ls", "userid", "startt", "endt", f"hc_{best_num}"]]
    df.rename(columns={f"hc_{best_num}": "cluster"}, inplace=True)
    df["cluster"] = df["cluster"].astype("int")

    # time
    df["startt"] = pd.to_datetime(df["startt"])
    df["endt"] = pd.to_datetime(df["endt"])
    df["dur_min"] = (df["endt"] - df["startt"]).dt.total_seconds() / 60

    # distance
    df["length_km"] = df["length_m"] / 1000

    # create the analysis folder
    curr_path = config["S_fig"] + f"\\{case}_5\\" + str(user)
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

    # organize modes
    ifOnlyWalk = df["mode_ls"].apply(_ifOnlyWalk)
    df.loc[ifOnlyWalk, "mode_encode"] = "Walk"
    df.loc[~ifOnlyWalk, "mode_encode"] = [_encode(i, mode_dict) for i in df.loc[~ifOnlyWalk, "mode_ls"].to_list()]

    ## visualize trip counts per class
    # trip count per cluster before filtering
    df.groupby("cluster").size().plot.bar(figsize=(10, 5))
    plt.xlabel(f"#Cluster {best_num}")
    plt.ylabel("Trip count")
    plt.savefig(curr_path + "\\num.png", bbox_inches="tight")
    plt.close()

    # filter classes, here >3% and not larger than 5
    df = df.groupby("cluster").apply(_filter, len(df)).dropna().reset_index(drop=True)
    if len(df["cluster"].unique()) > 5:
        top5 = df["cluster"].value_counts().head(5).index
        df = df.loc[df["cluster"].isin(top5)]

    df["cluster"] = df["cluster"].astype("int")
    # trip count per cluster after filtering
    df.groupby("cluster").size().plot.bar(figsize=(10, 5))
    plt.xlabel(f"#Cluster {best_num}")
    plt.ylabel("Trip count")
    plt.savefig(curr_path + "\\num_filtered.png", bbox_inches="tight")
    plt.close()

    end_period = datetime.datetime(2017, 12, 25)
    df = df.loc[df["endt"] < end_period]

    # mode frequency plot per cluster
    mode_freq = df.groupby("cluster").apply(_mode).dropna().reset_index()
    if len(mode_freq.columns) != 2:
        mode_freq.columns = ["clusters", "mode", "count"]
        _draw_freq(mode_freq)
        plt.savefig(curr_path + "/freq.png", bbox_inches="tight")
        plt.close()

    # 'length' and 'duration' scatter plot
    _draw_stat(df)
    plt.savefig(curr_path + "/stat.png", bbox_inches="tight", dpi=600)
    plt.close()

    #
    _draw_relative(df)

    # absolute evolution
    # u_all_trip = trips_df.loc[trips_df["userid"] == user]
    # _draw_absolute(df, u_all_trip)
