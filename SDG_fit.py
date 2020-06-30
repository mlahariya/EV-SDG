# ---------------------------------------------- ML 20/04/2020 -----------------------------------------------------#
#
# Here we create the SDG models and save them. This is dependent on specific arrival time model and departure and
# energy required
#       AM: Arrival model. This is generated using the fit_models script.
#           Required: slotted data file. prepared from the transactions file.
#                       EP ~ mean, loess, poly
#                       PP ~ neg_bio_reg, poisson_fit
#       MMc/MMe: mixture models for connection time and required energy.
#           Required: clustered data file. this is generated from the transcations data, which has,
#                     session and pole clusters. these are inputs for the mixtures.
#
#   IMP: run SDG_fit before running this file
#
#   dependencies: modelling.fit_models.py. This file implements the fitting for different models
#                 preprocess.create_slot_data.py.  to covert transactions data file to slotted data file.
#                 preprocess.prepare_clustered_data.py to generate session clusters, which will be used to create MMc and MMe
#                 modelling.data_hand.import_data.py used to transfer files from preprocess to modelling res, and also
#                   combine the session and pole cluster and ts data.
#                 modeling.data_hand.py for multiple extra functions to do operations on model/data
# -------------------------------------------------------------------------------------------------------------------- #

import argparse
import pickle
import datetime
import os
import sys
import numpy as np
import csv
# if needed
import warnings
warnings.filterwarnings("ignore")
# load the config file
import json

def main(args):
    # update the verbose of the fitting environment
    config = json.load(open('config.json'))
    config['verbose'] = int(args['verbose'])
    json.dump(config, open('config.json', 'w'))

    # Arguments
    config = json.load(open('config.json'))            # IMP.
    model_dump_loc = config['dir_names']['models_folder_name']
    model_log_dump_loc = config['dir_names']['models_logs_folder_name']
    process = str(args['model'])
    lambda_mod = str(args['lambdamod'])
    dump_filename = 'SDG Model ('+str(process)+","+str(lambda_mod)+') DT=' + str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M"))

    # ======================================= MODELS =================================================
    from modeling.fit_models import get_trained_model
    # Get the AM model
    # we need to have slotted data before we create a Arrivals model.
    # this part of code will pick up the transactions file from /res and create a slotted data file.
    # this slotted data file will be saved in the folder /preprocess. We will use this file to generate
    # our AM model. For further details, please look into preprocess.create_slot_data.py
    # if that file is already present, the code will be skipped.
    AM = get_trained_model(model="AM",process=process,lambda_mod=lambda_mod)
    # get the MMc and MMe models

    # import data from preprocess to the modeling folder
    # the years go as a list because we create clusters based on years. each year has seperate cluster files
    # if we put force = True for import_data, it will create the cluster files even if they are not present.
    # force = false, only works when clusterd files are already present for the given year
    # generate processed data
    MMc = get_trained_model(model="MMc")
    MMe = get_trained_model(model="MMe")

    # ======================================= SAVING =================================================
    from handles.model_hand import remove_raw_data
    from handles.model_hand import get_model_log
    SDG_model = list([AM, MMc, MMe])
    SDG_model_logs = [get_model_log(m) for m in SDG_model]
    empty_dict = dict.fromkeys(np.arange(1, 20))
    SDG_model_logs_dict = {**SDG_model_logs[0], **empty_dict, **SDG_model_logs[1]}
    SDG_model_logs_dict = {**SDG_model_logs_dict, ** empty_dict, ** SDG_model_logs[2]}
    with open(os.path.join(model_log_dump_loc,dump_filename+'.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in SDG_model_logs_dict.items():
            writer.writerow([key, value])

    # here we dump the data/models as a pickle file. which can be loaded later to be used.
    # We remove the raw data that was used to train the models and only save the fitted models i.e. _best_fit
    SDG_model = list([AM,MMc,MMe])
    SDG_model = [remove_raw_data(m) for m in SDG_model]
    filename = os.path.join(model_dump_loc,dump_filename)
    with open(filename, 'wb') as f:
        pickle.dump(SDG_model, f)
    sys.modules[__name__].__dict__.clear()

    print(" SDG Model saved at : ", model_dump_loc)
    print(" SDG Model File name : ", dump_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to SDG fit: \n '
                                                 'Creates a SDG model and saves it')
    parser.add_argument('-model', default='IAT',
                        help='modeling method to be used for modeling arrival times: \n'
                             '\t\t AC for arrival count models \n'
                             '\t\t IAT for inter-arrival time models')
    parser.add_argument('-lambdamod', default='mean',
                        help='Method to be used for modeling lambda:\n'
                             '\t\t AC: has two options, poisson_fit/neg_bio_reg \n'
                             '\t\t IAT: has three options, mean/loess/poly')
    parser.add_argument('-verbose', default=3,
                        help='0 to print nothing; >0 values for printing more information. Possible values:0,1,2,3 (integer)')

    args = parser.parse_args()
    main(vars(args))
