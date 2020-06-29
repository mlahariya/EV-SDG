# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# We create monthly clusters for sessions using this mydbscan file
#
# -------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

# load the config file
import json
config = json.load(open('config.json'))




class mydbscan:
    # this class is used to implement dpscan for us

    def __init__(self, epsilon, min_points, alpha):
        # Epsilon is the epsilon for the dbscan. alpha is the decrement that
        # will be done in epsilon to reach the given number of clusters.
        self._ep = epsilon
        self._minpts = min_points
        self._dbscan = []
        self._monthly_clusters = []
        self._alpha = alpha
        self._monthly_eps = []

    def data(self,data, norm=True):
        # Input data should be just columns of the points that need to be clustered. nothing else.

        if norm:
            # Scaling the data to bring all the attributes to a comparable level. Converting the numpy array into a
            # pandas DataFrame
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            norm_data = normalize(scaled_data)
            self._data = pd.DataFrame(norm_data)
        else:
            self._data = pd.DataFrame(data)


    def create_clusters(self,how_many = 4):
        # This function creates the DB scan of the points df = self._data.as_matrix().astype("float64", copy=False)
        # dist_matrix = sklearn.metrics.pairwise.euclidean_distances(df, df) We force each month to have a given
        # number of clusters, i.e. how_many clusters. also, after the clustring is over, we alsorecord the epsilon
        # value for that month to create that many clusters.
        for i in range(1,13):
            # Create a monthly data, generate its clusters and add them to the monthly clusters file
            month_data = self._data[self._data.Start_month == i].copy()
            month_data1 = month_data[['Start_time', "Departure_time"]]
            if config['verbose'] > 2: print(" \t\t Clustering for month :",i)
            epsilon = self._ep

            # Here we force all months to have the same number of clusters
            while True:
                db_created = DBSCAN(eps=epsilon, min_samples=self._minpts).fit(month_data1)
                epsilon = epsilon-self._alpha
                if config['verbose'] > 2: print(" \t\t Clusteres Created :",np.unique(db_created.labels_))
                if len(np.unique(db_created.labels_)) == how_many or epsilon < 0:
                    if epsilon < 0:
                        if config['verbose'] > 2: print(" \t\t ","Specified number of clusters not found -- ")
                    if len(np.unique(db_created.labels_)) == how_many:
                        if config['verbose'] > 2: print(" \t\t ","3 clusters found -- ")
                    self._monthly_eps.append([i,epsilon])
                    break

            self._dbscan.append(db_created)
            month_data['Clusters']=db_created.labels_
            self._monthly_clusters.append(month_data[['index','Clusters']])

    def plot_clusters(self,save=False,save_name=None):
        # Used to plot the points
        y_pred = self._dbscan.fit_predict(self._data)
        plt.scatter(self._data.iloc[:, 0], self._data.iloc[:, 1], c=y_pred, cmap='Paired',s=0.2)
        plt.title("DBSCAN")
        if save:
            plt.savefig(save_name)
        return plt



