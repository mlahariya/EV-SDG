# -------------------------------------------------- ML 22/04/2020 ----------------------------------------------------
#
# Here we generate the samples from the models.
# Please pass the AM,MMc and MMe as arguments to this file. Also, the save_loc and save_name can be used to tell
# where to save the sample data file.
#       - Each model. (AM,MMc and MMe) has a predict_day method.
#               This method can be called to generate data for a given type of day.
#               please read AM_analysis.py and MM_analysis.py for further details.
# --------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from handles.data_hand import create_factor_arr

import json
config = json.load(open('config.json'))


def generate_sample(AM,MMc,MMe,horizon_start,horizon_end):
    # create the factor for which data has to be generated.
    dates = pd.date_range(start=horizon_start, end=horizon_end)
    year = pd.Series( dates.year )
    month = pd.Series( dates.month )
    day = dates.dayofweek
    day = pd.Series( [0 if d < 5 else 1 for d in day ] )
    fac = create_factor_arr(year=year,month=month,daytype=day)


    # here we generate the data for the given horizon
    all_pred = pd.DataFrame()
    N_samps = fac.size
    number_of_wd = N_samps
    start = 0.05
    for i in range(0, number_of_wd):
        if config['verbose'] > 1: print( ' ------------------- Generating data for {} ----------------'.format(str(dates[i].date())))
        X_test = pd.DataFrame([fac[i]], columns=['Factor1'])

        # IMP = slot in the predict day method means slot seconds. meaning, how many seconds in each slot
        arrivals, t_next,_ = AM.predict_day(X_test=X_test, Start_time=np.array(start), slot=int(AM.slotmin)*60,
                                            variablity_lambda=True, verbose=config['verbose'])
        arrivals = arrivals.reshape(-1, 1)
        deps = MMc.predict_day(X_test=X_test, arrivals=np.array(arrivals), slot=int(AM.slotmin)*60, verbose=config['verbose'])
        deps = deps.reshape(-1, 1)
        energy_req = MMe.predict_day(X_test=X_test, arrivals=np.array(arrivals), slot=int(AM.slotmin)*60, verbose=config['verbose'])
        energy_req = energy_req.reshape(-1, 1)

        # prepare the data for saving
        ts = np.concatenate((np.repeat(dates[i].date(),arrivals.size).reshape(-1,1) ,arrivals,deps,energy_req) , axis=1).reshape(-1, 4)  # this is the number of predicted day
        predicted_days = ts
        predicted_days = pd.DataFrame(np.vstack(predicted_days))
        all_pred = all_pred.append(predicted_days)
        # t_next is the next days value
        start = t_next - 24
    # add model parameters
    all_pred.columns = ['Date','Arrival','Connected_time','Energy_required']
    all_pred['AM_specs'] = 'Arrival model {mod=' + str(AM.lambda_mod) +' variablity=' +  str(AM._variablity_lambda) + '}'
    all_pred['MMc_specs'] = 'Connected time model {mix=' + str(MMc._mix) +' method=' +  str(MMc._method) + '}'
    all_pred['MMe_specs'] = 'Energy Required model {mix=' + str(MMe._mix) +' method=' +  str(MMe._method) + '}'

    return all_pred