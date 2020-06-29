# -------------------------------------------------- ML 19/04/2020 ----------------------------------------------------
#
# This script can be called from the base dir. We fit the all different kinds on models here and return them for
# saving.
# Here we implement the three models for arrivals departure and energy.
#       Properties of Arrival models: Current implementation loess
#           2 types = exponential process/poisson process
#                  use combine to give the number of timeslots to be combined
#                  exponential process
#                       - lambda model - mean/poly/loess
#                       - use variability_lambda = Ture to control the monthly variance
#                       - sugggested to use log=Ture normal = true in case of poly/loess
#                       -
#                   poisson process
#                       - model - poisson/neg_bin/gamma
#      Properties of mixture models: for departure and energy required
#                   Current implementation GMMs (mixture of normals)
#                       - use inti for intilization. by default the session clusters are used for init
#                       - other implementation: Normal + Dmin/Beta + Dmin
#
#  Dependencies:
#           modelling: multiple scripts used to implement process
# --------------------------------------------------------------------------------------------------------------------


import argparse
import numpy as np
from modeling.stat.poles_selector import poles_selector
from modeling.stat.poisson_process import poisson_process
from modeling.stat.exponential_process import exponential_process
from modeling.stat.mixturemodels import mixture_models
from handles.data_hand import get_csv_data, create_factor_arr,get_slotted_data
import sys

# load the config file
import json
config = json.load(open('config.json'))

def get_trained_model(model,process="EP",lambda_mod="mean"):

    year = config['Year']
    SLOT = config['slot_mins']
    model = str(model)
    process = str(process)
    lambda_mod = str(lambda_mod)

    slot_sec = 60 * SLOT
    factor = 'Factor' # this is the name of factor we use to generate indipendent models of. usually of the form of
                       # year_month_daytype. we will have indipenden AM,MMc and MMe for these

    # --------------------------------------- LOADING DATASET ------------------------------------------------------
    if model == "AM":
        file_loc = config['filenames']['slot_data_file_path']
        file_name = config['filenames']['slot_data_file_name']
        required_file_name = "Slot_" + str(SLOT) + "_min_trans_data.csv"
    else:
        file_loc = config['filenames']['processed_data_file_path']
        file_name = config['filenames']['processed_data_file_name']
        required_file_name = 'Processed_' + str(SLOT) + '_min_' + str(year) + '_year_trans_data.csv'

    try:
        if config['verbose'] > 0: print(' ------------------- Training ',str(model),' model -------------------')
        all_data = get_csv_data(filename=file_name, filepath=file_loc)
    except:
        if config['verbose'] > 0: print(' ------------------- Required data file not found -------------------')
        if config['verbose'] > 0: print(' \t\t Please run SDG_preprocessing.py before this script')
        if config['verbose'] > 0: print(' \t\t Missing data file :',required_file_name)
        sys.exit("ERROR")
    # --------------------------------------- FIT and RETURN MODEL DATASETS -----------------------------------------
    # we have three different types of models that can be returned here
    #   AM = arrival model. The standard parameters here are of
    #       exponential process with variability lambda = true.
    #   MMc = mixture models for connection times
    #
    #   MMe = mixture model for required energy

    if model == 'AM':
        # --------------------------------------- PREPARE TRAINING DATASETS --------------------------------------------
        # Sorting data into required parameters
        all_data = all_data.sort_values(by=['Start_year', 'Start_DOY', 'Start_time']).copy()
        y_train = [2015]
        n_poles_test = [1677]

        # ------------------------------------  poles selection
        # This is a pole selector. given methods are onceeachn, topn and continous. for continous give all data
        PS = poles_selector(alldata=all_data,year=y_train)
        PS.select_poles(by='continous')
        charge_points = PS._charge_points
        # ------------------------------------- Training dataset
        ts_d = all_data[all_data['Start_year'].isin(y_train)]
        if config['verbose'] > 0: print(' \t\t Training AM for year: ' + str(year))
        if config['verbose'] > 0: print(' \t\t Training AM for slot minutes: ' + str(SLOT))
        if config['verbose'] > 0:print(' \t\t Total number of poles: ' + str(len(np.unique(ts_d['Charge_point']))))
        if config['verbose'] > 0:print(' \t\t Number of poles used:' + str(len(charge_points)))
        ts_d = ts_d[ts_d['Charge_point'].isin(charge_points)]
        n_poles_train = len(np.unique(ts_d['Charge_point']))
        ts_d = ts_d.reset_index()

        # --------------------------------- add the factor column to the dataset
        Start_times_slot = get_slotted_data(ts_d['Start_time'], slot_sec)
        ts_d['Start_time_slot'] = Start_times_slot
        weekday = ts_d['Start_weekday'].copy()
        weekday[ts_d['Start_weekday'] < 5] = 0
        weekday[ts_d['Start_weekday'] >= 5] = 1
        ts_d[factor] = create_factor_arr(year=ts_d['Start_year'], month=ts_d['Start_month'], daytype=weekday)

        # prepare the ts and x which are inputs for the exponential processes and poisson process
        # prepare the time seires
        ts = ts_d['Start_time'].copy()
        doy = ts_d['Start_DOY'].copy()
        sesonality = 24.00
        d = 1
        if config['verbose'] > 0: print(' \t\t Preparing time seires for modeling ... ')
        for i in range(1, ts.size):
            if doy[i] != doy[i - 1]:
                ts[i:] = ts[i:] + sesonality
                d = d + 1

        # prepare the X for generating TS
        x = ts_d[[factor]].copy()
        # preparing start times
        x['Start_time'] = ts_d[['Start_time_slot']].copy()
        x['Start_DOY'] = ts_d[['Start_DOY']].copy()

        if config['verbose'] > 0: print(' \t\t Training ... ')
        if process == "IAT":
            # model:
            # EXPONENTIAL PROCESS MODEL
            ep = exponential_process(events=ts, x=x, variablity_lambda=True, log=True, normalize=True)
            ep.fit(lambda_mod=lambda_mod, combine=np.arange(1, 7), poly_deg=1, alpha=0.125, max_poly_deg=30,
                   verbose=config['verbose'])
            if config['verbose'] > 0: print(' \t\t Trained ... ')
            return ep
        if process == "AC":
            # POISSON PROCESS MODEL
            pp = poisson_process(events=ts, x=x,variablity_lambda=True)
            pp.fit(lambda_mod=lambda_mod, combine=None,
                   verbose=config['verbose'])
            if config['verbose'] > 0: print(' \t\t Trained ... ')
            return pp

    # both MMc and MMe are idential except for the departure and energy columns.
    if model == 'MMc':
        if config['verbose'] > 0: print(' \t\t Training MMc for year: ' + str(year))
        if config['verbose'] > 0: print(' \t\t Training MMc for slot minutes: ' + str(SLOT))
        # --------------------------------------- PREPARING DATASETS -----------------------------------------------
        # Sorting data into required parameters
        all_data = all_data.sort_values(by=['Start_year', 'Start_DOY', 'Start_time']).copy()
        ts_d = all_data.copy()

        useful_data = ts_d[
            ['Start_time', 'Connected_time', 'Final_clusters', 'Final_Pole_clusters', 'Start_daytype', 'Factor']]
        useful_data['Start_time_slot'] = np.floor(useful_data['Start_time']) + 1

        # model
        # GMM for the mixture
        normal_mm_EM = mixture_models(y=useful_data['Connected_time'],
                                      x=useful_data[['Factor', 'Start_time_slot']],
                                      initilizations=useful_data['Final_clusters'],
                                      combine=None)
        normal_mm_EM.fit(mix='normal', method='EM', verbose=config['verbose'] )

        return normal_mm_EM

    if model == 'MMe':
        if config['verbose'] > 0: print(' \t\t Training MMe for year: ' + str(year))
        if config['verbose'] > 0: print(' \t\t Training MMe for slot minutes: ' + str(SLOT))
        # --------------------------------------- PREPARING DATASETS -----------------------------------------------
        # Sorting data into required parameters
        all_data = all_data.sort_values(by=['Start_year', 'Start_DOY', 'Start_time']).copy()
        ts_d = all_data.copy()

        useful_data = ts_d[
            ['Start_time', 'Energy_required', 'Final_clusters', 'Final_Pole_clusters', 'Start_daytype', 'Factor']]
        useful_data['Start_time_slot'] = np.floor(useful_data['Start_time']) + 1

        # model
        # GMM for the mixture
        normal_mm_EM = mixture_models(y=useful_data['Energy_required'],
                                      x=useful_data[['Factor', 'Start_time_slot']],
                                      initilizations=useful_data['Final_clusters'],
                                      combine=None)
        normal_mm_EM.fit(mix='normal', method='EM', verbose=config['verbose'] )

        return normal_mm_EM




