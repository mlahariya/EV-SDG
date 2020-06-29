# ---------------------------------------------- ML 25/10/2019 -----------------------------------------------------#
#
# This script creates the processed 15 minute transaction data. This data includes all the preprocessed information
# Including but not limited to
#   - Clusters : Session and poles
#   - Weekly/Weekday information ; Each slot information for poles
#   - Arrivals and departures of each transaction
# This file is called from SDG_preprocessing.py
#
# -------------------------------------------------------------------------------------------------------------------- #

import os
from handles.data_hand import get_csv_data
from preprocess.clustering.session_clusters import sesssion_clustering
from preprocess.clustering.pole_clusters import pole_clustering
from preprocess.create_slot_data import create_slot_data

# load the config file
import json
config = json.load(open('config.json'))

def create_processed_data():

    # Name and location of the final saved file
    save_name = 'Processed_' + str(config['slot_mins']) + '_min_' +str(config['Year'])+ '_year_trans_data.csv'
    save_loc = os.path.join(os.getcwd(), config["dir_names"]["preprocess_folder_name"])

    if os.path.exists(os.path.join(save_loc,save_name)):
        # if the data is already generated, then we dont need to worry
        if config['verbose']>0: print(' ------------------- Processed Data File Exists -------------------')

    else:
        if config['verbose'] > 0: print(' ------------------- Creating Processed Data File -------------------')

        # call slotting script. this will create the slotted data that we need from transactions
        slot_file_name,slot_file_loc = create_slot_data()

        # call session clustering script. This will generate the session clusters
        ses_clust_file_name, ses_clust_file_path = sesssion_clustering(slot_file_path=slot_file_loc,
                                                                       slot_file_name=slot_file_name)

        # call pole clustering script. This will generate the clusters for pole types
        pole_clust_file_name, pole_clust_file_path = pole_clustering(ses_clust_file_path=ses_clust_file_path,
                                                                    ses_clust_file_name=ses_clust_file_name)

        # Getting the session/pole clusters - Combining the data, and saving it.
        pole_clusts = get_csv_data(filename=pole_clust_file_name,
                                   filepath=pole_clust_file_path)
        session_clusts = get_csv_data(filename=ses_clust_file_name,
                                   filepath=ses_clust_file_path)
        fin_clust_data = session_clusts.join(pole_clusts.set_index('Charge_point'), on='Charge_point')
        fin_clust_data = fin_clust_data.drop(columns='index')
        fin_clust_data.reset_index()
        wd = fin_clust_data[['Start_weekday']]
        wd[wd < 5] = 0
        wd[wd >= 5] = 1
        fin_clust_data['Start_daytype'] = wd
        fin_clust_data['Factor'] = fin_clust_data['Start_year'].map(str) + '_' + \
                                    fin_clust_data['Start_month'].map(str) + '_' + \
                                    fin_clust_data['Start_daytype'].map(str)
        fin_clust_data.to_csv(os.path.join(save_loc, save_name))

        if config['verbose'] > 0: print(' Final clustering data file saved as :', os.path.join(save_loc,save_name))

        # update the names of files in configration files
        config['filenames'] = {}
        config['filenames']['slot_data_file_name'] = slot_file_name
        config['filenames']['ses_data_file_name'] = ses_clust_file_name
        config['filenames']['pole_data_file_name'] = pole_clust_file_name
        config['filenames']['processed_data_file_name'] = save_name
        config['filenames']['slot_data_file_path'] = slot_file_loc
        config['filenames']['ses_data_file_path'] = ses_clust_file_path
        config['filenames']['pole_data_file_path'] = pole_clust_file_path
        config['filenames']['processed_data_file_path'] = save_loc
        json.dump(config, open('config.json', 'w'))

    return save_name, save_loc

