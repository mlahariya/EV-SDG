# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# This file creates the final clusters for sessions. We create monthly clusters first, combine them, and then
# associate noise points with the nearest cluster.
#
# This script is called by prepare_clustered_data.py
#
#
# Support Script - preprocess/clustering/monthly_cluster_data_point.py
# -------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
from handles.data_hand import get_csv_data
import os
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import ConvexHull

# load the config file
import json
config = json.load(open('config.json'))

# The main function.
def sesssion_clustering(slot_file_name,slot_file_path):

    Year = config['Year']
    slot_file_name = slot_file_name
    slot_file_path = slot_file_path
    save_name = "Final_session_clustered_" + str(Year) + "_trans_data.csv"
    save_location = os.path.join(os.getcwd(), config["dir_names"]["preprocess_folder_name"], 'session_cluster')
    monthclust_file = 'Monthly_clustered_' + str(Year) + '_trans_data.csv'
    monthclust_filepath = os.path.join(os.getcwd(), config["dir_names"]["preprocess_folder_name"], 'session_cluster')


    # Check if the final clustered file exists or not
    if os.path.exists(os.path.join(save_location, save_name)):
        if config['verbose'] > 1:
            print(' ------------------- Annual clusters exist ------------------------')
            print(" \t\t Final Clustered Data Saved as: ",
                  os.path.join(save_location, save_name))
            print(' \t\t Clusters Created for year :' + str(Year))
        return save_name, save_location
    else:

        # ------------------------------------------ Monthly Clusters --------------------------------------------------
        # Check if monthly clusters are created. If not, create them

        if os.path.exists(os.path.join(monthclust_filepath, monthclust_file)):
            if config['verbose'] > 1: print(' ------------------- Monthly Cluster File Exists --------------------')
        else:
            if config['verbose'] > 1: print(' ------------------- Creating Monthly Cluster File --------------------')
            from preprocess.clustering.monthly_cluster_data_points import main as monthly_ses_clust
            monthly_ses_clust(slot_file_name=slot_file_name,
                              slot_file_path=slot_file_path,
                              save_loc = monthclust_filepath,
                              save_name = monthclust_file)
            if config['verbose'] > 1: print(" Monthly clustered Data Saved as: ",
                                         os.path.join(monthclust_filepath, monthclust_file))

        # ----------------------------------- Monthly Clusters data Loading  -------------------------------------------
        X = get_csv_data(filename=monthclust_file, filepath=monthclust_filepath)

        # ANNUAL CLUSTERS ------------------------------------------------------- Here we prepare a annual
        # clustering. we calculate the means of monthly clusters, and then cluster these means. depending on this new
        # clustering, we add the final_clusters column in the X_annual data set. We fill the noise with -1 factor We
        # also plot the annual clustered points
        means = X.groupby(['Start_month', 'Clusters'], as_index=False).agg(
            {'Start_time': ['mean'], 'Departure_time': ['mean']})
        means.columns = ['Start_month', 'Clusters', 'Start_mean', 'Departure_mean']
        means_withoutnoise = means[means['Clusters'] >= 0]
        mean_clusters = DBSCAN(eps=2, min_samples=6).fit(means_withoutnoise[['Start_mean', 'Departure_mean']])
        means_withoutnoise['Final_clusters'] = mean_clusters.labels_
        means_withoutnoise = means_withoutnoise.copy()
        X_annual = pd.merge(X, means_withoutnoise, on=['Start_month', 'Clusters'], how='outer').fillna(-1)

        # ------------------------------------------ NOISE REMOVAL ----------------------------------------------------
        # Now we include the noise in the clusters too. here we classify each noise point to a given cluster point,
        # based on its distance from the nearest clustered point. This seems very time consuming. We find dist of all
        # points wrt to one point, find the min dist point that is clustered and the assign this point to that cluster.
        if config['verbose'] > 1: print(' ------------------- Processing Noise Data Points ------------------- ')
        bool = X_annual[['Final_clusters']] < 0
        indexes = np.where(bool)[0]
        Final_clusters = X_annual[['Final_clusters']]
        noise_ind = np.where( X_annual[['Final_clusters']] < 0)[0]
        data_ind = np.where(X_annual[['Final_clusters']] >= 0)[0]
        boundry_points = X_annual.iloc[data_ind,:]
        boundry = []
        for i in np.unique(boundry_points[['Final_clusters']]):
            data = boundry_points[boundry_points['Final_clusters'] == i]
            data = np.array(data[['Start_time', 'Departure_time']]).reshape(-1,2)
            hull = ConvexHull(data)
            bp = np.insert(data[hull.vertices],2,i,axis=1)
            boundry = np.append(boundry,bp)

        # boundry of the clusters
        boundry = boundry.reshape(-1,3)

        dist_matrix = pairwise_distances(np.array(X_annual.iloc[noise_ind,:][['Start_time', 'Departure_time']]),
                                         boundry[:, :2])

        min_indexes = np.argmin(dist_matrix, axis=1)
        clusts = np.array([boundry[i,2] for i in min_indexes])
        Final_clusters.iloc[indexes] = clusts.reshape(-1,1)

        X_annual[['Final_clusters']] = Final_clusters.copy()

        # SAVING -------------------------------------------------------
        X_annual.to_csv(os.path.join(save_location, save_name), index=False)
        if config['verbose'] > 1: print(" Session clusters data saved as: ",
              os.path.join(save_location, save_name))
        if config['verbose'] > 1: print(' \t\t Clusters Created for year :' + str(Year))

        if config['create_plots']:
            # create plots
            colors = np.array(X_annual[['Final_clusters']])
            plt.scatter(X_annual[['Start_time']], X_annual[['Departure_time']],
                        c=colors, cmap='Paired', s=0.2)
            plt.savefig(os.path.join(save_location, 'Final_session_clust_' + str(Year) + '_plot.png'))

        return save_name, save_location

