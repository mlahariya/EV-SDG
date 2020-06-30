# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# We create pole clusters here. Once the pole clusters are created, we add them to the final transaction data
#
# This script is called by prepare_clustered_data.py
# dependencies: preprocess.clustering.pole_class
#
# -------------------------------------------------------------------------------------------------------------------- #

# Import required libraries
from preprocess.clustering.pole_class import pole
from functools import reduce
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# load the config file
import json
config = json.load(open('config.json'))

def point_loc(x,y,z):
    val = 0*x + 0.5*y - 0.5*z
    if val > 0:
        return True
    else:
        return False

# The main function.
def pole_clustering(ses_clust_file_path,ses_clust_file_name):
    Year = config['Year']
    annual_sessions_cutoff = 0                                  # Cutoff for the minimum number of sessions per year
    Combine_start_slots = 4                                     # how many time slots to combine while plotting
    Clustering_on = 'Percentage'                                # what to cluster with
    Cluster_by = 'Session_clusters'                             # 'Which parameters to cluster the data by:
                                                                # Session_clusters, Day_type, Time_slot, All'

    finalclust_file = ses_clust_file_name
    file_path = ses_clust_file_path
    save_location = os.path.join(os.getcwd(), config["dir_names"]["preprocess_folder_name"], 'pole_cluster')
    save_name = "Poles_clustered_" + str(Year) + "_by" + Cluster_by + "_cutoff " + str(
        annual_sessions_cutoff) + ".csv"

    if os.path.exists(os.path.join(save_location,save_name)):
        if config['verbose'] > 1: print(' ------------------- Pole Cluster File Exists --------------------')
        return save_name,save_location
    else:
        # CREATING POLES CLUSTERS ---------------------------------------------------------------------------------
        if config['verbose'] > 1: print(' ------------------- Creating Pole Cluster File --------------------')
        pole_class = pole(Clustered_filename=finalclust_file, Clustered_filepath=file_path)
        pv = pole_class.pivot_poles(col='Day_type')
        pv1 = pole_class.pivot_poles(col='Start_time_slot')
        pv2 = pole_class.pivot_poles(col='Final_clusters')
        pv2.columns = ['Charge_point','C0','C1','C2']
        pole_properties = reduce(lambda x,y: pd.merge(x,y, on='Charge_point', how='outer'), [pv, pv1, pv2])

        # Remove less than 200 samples per year
        indexer = np.array( pv[['Weekday']] ) + np.array( pv[['Weekend']])
        indexer = indexer>annual_sessions_cutoff
        pv = pv[indexer]
        pv1 = pv1[indexer]
        pv2 = pv2[indexer]
        pole_properties = pole_properties[indexer]

        # Here is the data to perform clustring: We have multiple options, however, we only choose to do it with
        # percentage of sessions in each session cluster
        if Cluster_by == 'Day_type':
            data_cluster = pv.drop('Charge_point', 1)
        if Cluster_by == 'Time_slot':
            data_cluster = pv1.drop('Charge_point', 1)
        if Cluster_by == 'Session_clusters':
            data_cluster = pv2.drop('Charge_point', 1)
            if Clustering_on == 'Percentage':
                data_cluster = data_cluster.div(data_cluster.sum(axis=1), axis=0)
                temp  = data_cluster.copy()
                temp['Charge_point']= pv2[['Charge_point']]
                pv2 = temp.copy()

        if Cluster_by == 'All':
            data_cluster = pole_properties.drop('Charge_point', 1)

        # # Clustering
        # # Loop to find best combinations ep and min points
        # ep = 0.05
        # while ep <= 0.5:
        #     min_samples = 5
        #     while min_samples <= 200:
        #         db = DBSCAN(eps=ep, min_samples=min_samples).fit(data_cluster)
        #         labels = db.labels_
        #         print('Epsilon :',ep,' ; Minimum Samples :',min_samples,' ; Clusters :',np.unique(labels))
        #         min_samples = min_samples + 5
        #     ep = ep + 0.05
        # # ABSOLUTES EP 40 and min_samples 10 - gives us 3 clusters
        # # Percentages EP 0.05 and min_samples 5/55 - gives us 4/5 clusters
        db = DBSCAN(eps=0.05, min_samples=55).fit(data_cluster)
        labels = db.labels_
        if config['verbose'] > 1: print(' \t\t Epsilon :',40,' ; Minimum Samples :',10,' ; Clusters :',np.unique(labels))
        pole_properties['Pole_clusters'] = labels

        # Eq of the plane
        # 0X + 0.5Y - 0.5Z + 0 = 0
        clusts = labels.copy()
        for i in range(0,labels.shape[0]):
            if point_loc(x = pv2.loc[i,'C0'],y= pv2.loc[i,'C1'], z = pv2.loc[i,'C2']):
                    clusts[i] = 1
            else:
                    clusts[i] = 2
        clusts[labels == 0] = 0
        pole_properties['Final_Pole_clusters'] = clusts
        pole_properties = pole_properties.drop(columns="Pole_clusters")

        # SAVING -------------------------------------------------------
        pole_properties.to_csv(os.path.join(save_location,save_name), index=False)
        if config['verbose'] > 1: print(" Pole clusters Data Saved as: ", os.path.join(save_location, "Poles_clustered_" + str(Year) + "_data.csv"))


        #
        # ---------------------------------------- PLOTTING ---------------------------------------------------------
        if config['create_plots']:
            # C0 == PARK TO CHARGE
            # C1 == CHARGING NEAR HOME
            # C2 == CHARGING NEAR WORK
            map = {-1: 'blue', 0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: 'orangered'}
            # plotting Clusters of start end time
            # old clusters
            temp = pv2.copy()
            temp['Pole_clusters'] = labels
            temp = temp.set_index('Charge_point')
            plt_d = pole_class.bplot(X='Final Clusters', Y='Number Starts', ID='Pole_clusters', data=temp)
            plt_d.savefig(
                os.path.join(save_location, "PoleC_VS_SessionC_bar_" + str(Year) + "_by" + Cluster_by + "_cutoff " + str(
                    annual_sessions_cutoff) + ".png"))
            plt_d.clf()

            # 3D plot
            fig = plt.figure()
            ax = Axes3D(fig)

            # Plot the values
            ax.scatter(temp[['C0']], temp[['C1']], temp[['C2']], c=temp['Pole_clusters'].apply(lambda x: map[x])
                       , s=2, marker='o')
            ax.set_xlabel('PARK TO CHARGE')
            ax.set_ylabel('CHARGING NEAR HOME')
            ax.set_zlabel('CHARGING NEAR WORK')
            ax.set_title("Poles distribution on Session Clusters")
            ax.view_init(azim=100)
            plt.savefig(os.path.join(save_location, "PoleC_VS_SessionC_Scatter_" + str(Year) + "_by" + Cluster_by + "_cutoff "
                                     + str(annual_sessions_cutoff) + ".png"))
            plt.clf()
            # plotting Clusters of start end time
            # updated clusters
            temp = pv2.copy()
            temp['Pole_clusters'] = clusts
            temp = temp.set_index('Charge_point')
            plt_d = pole_class.bplot(X='Final Clusters', Y='Number Starts', ID='Pole_clusters', data=temp)
            plt_d.savefig(
                os.path.join(save_location, "Fin_PoleC_VS_SessionC_bar_" + str(Year) + "_by" + Cluster_by + "_cutoff " + str(
                    annual_sessions_cutoff) + ".png"))
            plt_d.clf()

            # 3D plot
            fig = plt.figure()
            ax = Axes3D(fig)
            # Plot the values
            ax.scatter(temp[['C0']], temp[['C1']], temp[['C2']], c=temp['Pole_clusters'].apply(lambda x: map[x])
                       , s=2, marker='o')
            ax.set_xlabel('PARK TO CHARGE')
            ax.set_ylabel('CHARGING NEAR HOME')
            ax.set_zlabel('CHARGING NEAR WORK')
            ax.set_title("Poles distribution on Session Clusters")
            ax.plot([1, 0], [0, 0.5], [0, 0.5], color='r')
            ax.view_init(azim=100)
            plt.savefig(os.path.join(save_location, "Fin_PoleC_VS_SessionC_Scatter_" + str(Year) + "_by" + Cluster_by + "_cutoff "
                                     + str(annual_sessions_cutoff) + ".png"))
            plt.clf()

            # plotting Day types. we also plot a scatter plot for Visulization
            pv['Pole_clusters'] = clusts
            pv = pv.set_index('Charge_point')
            plt_d = pole_class.bplot(X='Day type',Y='Number Starts',ID='Pole_clusters', data=pv)
            plt_d.savefig(
                os.path.join(save_location, "PoleC_VS_daytype_bar_" + str(Year) + "_by" + Cluster_by + "_cutoff " + str(
                    annual_sessions_cutoff) + ".png"))
            plt_d.clf()
            colors = np.array(pv[['Pole_clusters']])
            plt.scatter(pv[['Weekday']], pv[['Weekend']],c=colors, cmap='Paired', s=2)
            plt.title("Poles distribution with type of day")
            plt.xlabel("WeekDay")
            plt.ylabel("WeekEnd")
            plt.savefig(os.path.join(save_location, "PoleC_VS_daytype_scatter_" + str(Year) + "_by" + Cluster_by + "_cutoff " + str(
                    annual_sessions_cutoff) + ".png"))
            plt.clf()


            # plotting starting times
            combine_slots = Combine_start_slots
            pv1 = pv1.set_index('Charge_point')
            pv1_combined = pd.DataFrame(np.add.reduceat(pv1.values, np.arange(len(pv1.columns))[::combine_slots], axis=1))
            pv1_combined.columns = pv1_combined.columns + 1
            pv1_combined.add_prefix('_Hour')
            pv1_combined['Pole_clusters'] = clusts
            plt_d.figure(figsize=(15,10))          # Width, height
            plt_d = pole_class.bplot(X='Start Time',Y='Number Starts',ID='Pole_clusters', data=pv1_combined)
            plt_d.savefig(os.path.join(save_location, "PoleC_VS_StartTimes_bar_" + str(Year) + "_by" + Cluster_by + "_cutoff " + str(
                    annual_sessions_cutoff) + ".png"))
            plt_d.clf()

        return save_name, save_location