# -------------------------------------------------- ML 11/12/2019 ----------------------------------------------------
#
# Here we predict the arrival times for the given series
#
# --------------------------------------------------------------------------------------------------------------------




import pandas as pd
import os
import argparse
import numpy as np
from modeling.stat.processes import exponential_process
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------------------------------- #
# Required functions
def get_csv_data(filename, filepath):
    # This function returns the data. data format should be in csv
    # read the raw data and prepare print out the first five rows of the data

    raw_data = pd.read_csv(os.path.join(filepath,filename))
    print("-------------- DATA FORMAT ------------------")
    print(raw_data.head())
    return raw_data



def get_slotted_data(data, slot_secs):
    # created slots are in form of ceilings.
    factor = slot_secs/3600
    # columns_to_divide = ['Start_time_slot', 'ChargeTime', 'ConnectedTime']
    data= ((data// factor)+1).astype(int)
    return data

def main(args):


    test_fac = (args['test_fac'])
    SLOT = int(args['slotmins'])
    TS_file_name = (args['TS_file_name'])
    TS_file_path = (args['TS_file_path'])
    factor = str(args['factor'])
    Continous_l = args['Continous_l']

    slot_sec = 60*SLOT
    save_loc = 'res\modeling'

    ts_d = get_csv_data(filename=TS_file_name,filepath=TS_file_path)

    # --------------------------------------- PREPARING DATASETS -----------------------------------------------
    # Sorting data into required parameters
    ts_d = ts_d.sort_values(by=['Start_year','Start_DOY','Start_time']).reset_index()

    # preparing values for plotting
    Start_times_slot = get_slotted_data(ts_d['Start_time'], slot_sec)
    ts_d['Start_time_slot'] = Start_times_slot
    plt_vals = ts_d.groupby(['Start_DOY', factor, 'Start_time_slot']).agg(
        {'Start_time': 'count'}).reset_index()
    plt_vals['Value'] = 'Actual'


    # prepare the time seires
    ts = ts_d['Start_time'].copy()#.head(1000)
    doy = ts_d['Start_DOY'].copy()#.head(1000)
    sesonality = 24.00
    d = 1
    print('Preparing time seires for modeling ... ')
    for i in range(1, ts.size):
        if doy[i] != doy[i - 1]:
            ts[i:] = ts[i:] + sesonality
            d = d + 1

    # prepare the X for generating TS
    x = ts_d[[factor]].copy()#.head(1000)
    # preparing start times
    x['Start_time'] = ts_d[['Start_time_slot']].copy()

    # ------------------------------------- MAKING PREDICTIONS AND PLOTTING -----------------------------------
    # fitting
    ep = exponential_process(events=ts, x=x)
    # max plot degree is the max poly degree that is tried and fitted
    # ep.fit(max_poly_deg=20,mod='binned_exp',bins=24*60*60)
    # iter is the number of iterations for which the process should be run

    cont = Continous_l
    ep.fit(mod='binned_exp',continous=cont,max_poly_deg=20)

    # save model
    model = pd.DataFrame(ep._best_fit)
    filename = factor + ' model (' + 'Slot_mins=' + str(SLOT) + ' Continous=' + str(cont) + ')'
    model.to_csv(os.path.join(save_loc,'model',filename + '.csv'))

    # make plots. here we plot our p val plots and save them
    name = 'Slot_mins=' + str(SLOT) + 'Continous=' + str(cont)
    result = ep.plt_model(name=name, loc=os.path.join(save_loc, 'model'), what='kstest')

    #  -------------------------------------------- PREDICTION ---------------------------------------------
    fac_levels = np.unique(ts_d[[factor]])

    all_pred = pd.DataFrame()

    N_samps = test_fac.size
    number_of_wd = N_samps
    start = 0.05
    for i in range(0, number_of_wd):
        X_test = pd.DataFrame([test_fac[i]], columns=[factor])
        # X_test = pd.DataFrame(x.iloc[0]).transpose()
        # X_test=X_test.drop(columns=['Start_time'])
        ts,t_next = ep.predict_day(X_test=X_test, Start_time=np.array(start), slot=slot_sec)
        ts = ts.reshape(-1, 1)
        ts = np.insert(ts, 1, i, axis=1).reshape(-1, 2) # this is the number of predicted day
        predicted_days = ts
        predicted_days = pd.DataFrame(np.vstack(predicted_days))
        predicted_days[factor] = test_fac[i]
        all_pred=all_pred.append(predicted_days)
        # t_next is the next days value
        start = t_next - 24


    # saving the files only if its asked to save the model
    filename = factor + ' predictions (#days sampled=' + str(N_samps) +'Slot_mins=' + str(SLOT) + 'Continous=' + str(cont)+')'
    all_pred.to_csv(os.path.join(save_loc,'arrivals',filename + '.csv'))




def run_script(test_fac,slotmins,ts_file_name,ts_file_path,factor,Continous_l=False):
    # argument praser that will be requiered for this file. defaults are set.
    parser = argparse.ArgumentParser(description='Arguments to create a slotted data')
    parser.add_argument('-TS_file_name', default=ts_file_name,
                        help='the name of raw data file')
    parser.add_argument('-TS_file_path', default=ts_file_path,
                        help='location of raw data file')
    parser.add_argument('-test_fac', default=test_fac,
                        help='array of factor values to generate results')
    parser.add_argument('-slotmins', default=slotmins,
                        help='minutes in each slot')
    parser.add_argument('-factor', default=factor,
                        help='factor for which to plot')
    parser.add_argument('-Continous_l', default=Continous_l,
                        help='if a confinous lambda should be fitted')

    args = parser.parse_args()

    main(vars(args))