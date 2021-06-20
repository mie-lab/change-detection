import pandas as pd
import numpy as np
import os
import multiprocessing
import pickle
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13

import scipy

from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

from utils.config import config
from similarityMeasures import getValidTrips

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def __k_Medoids(data, trip_df, bound=[3, 20], repeat=10):
    """__k_Medoids clustering algorithm"""

    print(data.shape)

    sil_list = []
    best_n = 0
    best_score = -1
    best_inertia = np.array([])

    for num_cluster in range(bound[0], bound[1]):

        sil_repeat_list = []
        best_repeat_score = -1
        best_repeat_labels = np.array([])
        best_repeat_inertia = np.array([])

        # repeat to find
        for _ in range(repeat):
            kmedoids = KMedoids(n_clusters=num_cluster, metric="precomputed", init="k-medoids++")
            kmedoids.fit(data)

            labels = kmedoids.labels_
            inertia = kmedoids.inertia_

            score = silhouette_score(data, labels, metric="precomputed")

            if score > best_repeat_score:
                best_repeat_score = score
                best_repeat_labels = labels
                best_repeat_inertia = inertia

            # add the current score
            sil_repeat_list.append(score)

        # save the labels with the best sil score
        trip_df[f"kc_{num_cluster}"] = best_repeat_labels

        # get the best score
        if best_repeat_score > best_score:
            best_n = num_cluster
            best_score = best_repeat_score
            best_inertia = best_repeat_inertia

        # add current score list
        sil_list.append(sil_repeat_list)

    # convert silhouette score to dataframe
    sil_df = pd.DataFrame(np.array(sil_list).T, columns=np.arange(bound[0], bound[1]))

    return trip_df, {"silhouette_df": sil_df, "best_inertia": best_inertia, "n_clusters": best_n}


def __save_results(cluster_df, r_dict, curr_path, bound, idx):
    """Result saving for __k_Medoids"""
    # plotting
    r_dict["silhouette_df"].max(axis=0).plot(label="")

    plt.title("Max silhouette score")
    plt.xlabel("# Clusters", fontsize=16)
    plt.ylabel("Silhouette index", fontsize=16)
    plt.xticks(np.arange(bound[0], bound[1]))
    plt.ylim([0, 1])
    # plt.legend(loc="upper right",prop={'size': 13})

    # save figure
    plt.savefig(curr_path + f"/{idx}_sil.png", dpi=600, bbox_inches="tight")
    plt.close()

    # save silhouette score
    r_dict["silhouette_df"].to_csv(curr_path + f"/{idx}_sil.csv")

    # save best result
    cluster_num = r_dict["n_clusters"]
    cluster_df[f"kc_{cluster_num}"].value_counts().to_csv(curr_path + f"/k_value_counts.csv")
    cluster_df.to_csv(curr_path + f"/{idx}_N{cluster_num}.csv", index=False)


def _index(mat, c_index):
    """index (WB and CH) for __hierarchy"""
    unique_clus = list(set(c_index))
    traj_num = len(c_index)
    clus_num = len(unique_clus)
    new_mat = 1 - mat
    np.fill_diagonal(new_mat, 0)

    ssw_ls = []
    idx_ls = []
    ssb = 0
    count = 0
    # SSW
    for clus in unique_clus:
        idx = np.where(c_index == clus)[0]
        idx_ls.append(idx)
        if idx.shape[0] == 1:
            count += 1
        ssw_ls.append(new_mat[idx[:, None], idx[None, :]].max())

    for i in range(clus_num):
        for j in range(i + 1, clus_num):
            curr_arr = new_mat[idx_ls[i][:, None], idx_ls[j][None, :]]
            ssb += curr_arr.min()

    ssw = max(ssw_ls) + count

    WB = clus_num * ssw / ssb
    CH = (ssb / (clus_num - 1)) / (ssw / (traj_num - clus_num))
    return WB, CH


# Hierarchical clustering algorithm
def __hierarchy(data, t_df, curr_path, bound, idx):

    # the distance criteria
    distance_criteria = "complete"

    # change the distance matrix to condense and cluster via linkage
    condense = scipy.spatial.distance.squareform(data)

    Z = linkage(condense, distance_criteria)

    # show the dendrogram
    plt.figure(figsize=(25, 10))
    # dendrogram(Z, p=50, truncate_mode = 'lastp', color_threshold=0.4*max(Z[:,2]), no_labels=True)
    dendrogram(Z, color_threshold=0.4 * max(Z[:, 2]), no_labels=True)
    plt.savefig(curr_path + f"/{idx}_{distance_criteria}_den.png")
    plt.close()

    best_score = 0
    sil_list = []
    # get the different cluster results
    for i in range(bound[0], bound[1]):
        cluster_result = fcluster(Z, t=i, criterion="maxclust")

        # use silhouette score as measure
        score = silhouette_score(data, cluster_result, metric="precomputed")

        if score > best_score:
            best_score = score

        # add the current score
        sil_list.append(score)
        # WB, CH = _index(mat, c_re)

        t_df[f"hc_{i}"] = cluster_result

    # plot showing the silhouette score for each cluster
    plt.plot(np.arange(bound[0], bound[1]), sil_list, label="")

    plt.title("Max silhouette score")
    plt.xlabel("# Clusters", fontsize=16)
    plt.ylabel("Silhouette index", fontsize=16)
    plt.xticks(np.arange(bound[0], bound[1]))
    plt.ylim([0, 1])
    # plt.legend(loc="upper right",prop={'size': 13})

    # save figure
    plt.savefig(curr_path + f"/{idx}_sil.png", dpi=600, bbox_inches="tight")
    plt.close()

    # save silhouette score
    pd.DataFrame(sil_list).to_csv(curr_path + f"/{idx}_sil.csv")

    # save best result
    best_cluster_num = np.arange(bound[0], bound[1])[np.argmax(sil_list)]
    t_df[f"hc_{best_cluster_num}"].value_counts().to_csv(curr_path + f"/h_value_counts.csv")

    t_df.to_csv(curr_path + f"/{idx}_N{best_cluster_num}.csv", index=False)


def cluster(dist, t_df, curr_path):
    """Clustering of trips based on the measured similarity."""

    # >20 might not be interesting for the study
    bound = [3, 21]
    ##########
    #### k_Medoids
    # feature_set = "K"
    # cluster_df, r_dict = __k_Medoids(dist, t_df.copy(), bound=bound, repeat=100)
    # __save_results(cluster_df, r_dict, curr_path, bound, feature_set)

    ##########
    # Hierarchical clustering
    feature_set = "H"
    __hierarchy(dist, t_df.copy(), curr_path, bound, feature_set)


if __name__ == "__main__":
    t_df = getValidTrips(time_window=5)

    ## open distance matrix
    with open(config["similarity"] + f"/similarity.pkl", "rb") as f:
        dist_mat = pickle.load(f)

    ## parallel clustering
    jobs = []
    for user in t_df["userid"].unique():

        # get the distance matrix
        dist = dist_mat[user]["all"]

        # get the trip ID
        ut_df = t_df.loc[t_df["userid"] == user]

        # create the user folder
        curr_path = config["cluster"] + f"\\{user}"
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)

        # perform clustering
        p = multiprocessing.Process(target=cluster, args=(dist, ut_df, curr_path))

        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

