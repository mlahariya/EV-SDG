# -------------------------------------------------- ML 30/9/2019 -----------------------------------------------------#
#
# This file is used to create the slotted data from transactions file obtained from Elaad.
# The output of the file is a slotted data file. there are following parameters in the output slotted file
#   - Date: in form of year/day/month/Day of year/weekday
#   - Arrival: in form of arrival time slot and arrival time
#   - Energy: In terms of energy required and energy required slot. this is the total energy in form of KWH
#   - Connection: In form of connected time and connected time slot
#
# THIS IS IMPORTANT:
#           HERE WE IMPLEMENT A DATA CLEANING STEP FOR OUR DATASET
#           IF YOU ARE USING A NEW DATASET, PLEASE CHANGE THE DATA CLEANING STEP AS NECESSARY
#
#
# This script is called by prepare_clustered_data.py
# -------------------------------------------------------------------------------------------------------------------- #

# Required libraries
import pandas as pd
import os
from handles.data_hand import get_csv_data

# load the config file
import json
config = json.load(open('config.json'))

# variables to be used in this file
# raw data location and file name. this is the prepared transactions csv file
# raw_filename = the name of the raw data file.
# raw_filepath = location of the raw data file
# slot_minutes = number of minutes in one slot
# save_loc = the location to save the slotted data file. by default it is saved in the current location
# These para meters are defined/defaulted in the configfile.


# ---------------------------------------------------------------------------------------------------- #
def get_date_time(data,datetime_col_name,suffix):
    # This function splits the datetime into dates and time slot columns
    # input suffix is added to the column names to keep track of which date are changed.

    df = pd.DataFrame()
    df['day'] = [d.day for d in pd.to_datetime(data[datetime_col_name])]
    df['month'] = [d.month for d in pd.to_datetime(data[datetime_col_name])]
    df['year'] = [d.year for d in pd.to_datetime(data[datetime_col_name])]
    df['DOY'] = [d.strftime('%j') for d in pd.to_datetime(data[datetime_col_name])]
    df['Weekday'] = [d.weekday() for d in pd.to_datetime(data[datetime_col_name])]
    df['time_slot'] = ( (pd.to_datetime(data[datetime_col_name]) -
                  pd.to_datetime(data[datetime_col_name]).dt.floor('D')).dt.total_seconds()  ) / 3600
    df.columns = [suffix+df.columns]
    return df

def get_processed_data(raw_data):
    # starting point of charging is included. Ending is commented out but can also be included.
    start = get_date_time(data=raw_data, datetime_col_name='Started', suffix='Start_')
    # end = get_date_time(data=raw_data, datetime_col_name='Ended',suffix='End_')

    # These are the energy requirment and departure times
    energy = raw_data['TotalEnergy']
    departure = raw_data['ConnectedTime']

    # pole data
    pole = raw_data['ChargePoint']

    # here we combine the data to form a processed dataframe
    p_data = pd.concat([start.reset_index(drop=True),
                        energy.reset_index(drop=True),
                        departure.reset_index(drop=True),
                        pole.reset_index(drop=True)], axis=1)
    return p_data

def get_slotted_array(data, slot_secs):
    # created slots are in form of ceilings.
    factor = slot_secs/3600
    # columns_to_divide = ['Start_time_slot', 'ChargeTime', 'ConnectedTime']
    data['Start_time_slot'] = ((data['Start_time'] // factor)+1).astype(int)
    data['Energy_required_slot'] = ((data['Energy_required'] // factor)+1).astype(int)
    data['Connected_time_slot'] = ((data['Connected_time'] // factor)+1).astype(int)
    return data

# ---------------------------------------------------------------------------------------------------- #
# The main function.
# this will pull the arguments, get the raw data, create a processed data and slotted data
# slots create are in the form of ceilings. i.e. for a 15 minute slots, 23 minutes is the 2nd slot.
def create_slot_data():
    raw_filename = config['transactions_filename']
    raw_filepath = os.path.join(os.getcwd(),config["dir_names"]["res_folder_name"])
    slot_minutes = config['slot_mins']
    save_location = os.path.join(os.getcwd(),config["dir_names"]["preprocess_folder_name"])
    save_file_name = "Slot_"+str(slot_minutes)+"_min_trans_data.csv"



    if os.path.exists(os.path.join(save_location,save_file_name)):
        if config['verbose'] > 1:
            print(' ------------------- Slotted data already present : Slots of ' + str(slot_minutes) +
                  ' minutes -------------------')
        return save_file_name,save_location

    else:

        # get raw data
        raw_data = get_csv_data(filename=raw_filename, filepath=raw_filepath)

        # process the raw data file
        processed_data = get_processed_data(raw_data=raw_data)
        processed_data.columns = ['Start_day', 'Start_month', 'Start_year', 'Start_DOY','Start_weekday', 'Start_time', 'Energy_required',
                                  'Connected_time', 'Charge_point']

        # create slotted data file from processed data file
        slotted_data = get_slotted_array(data=processed_data, slot_secs=slot_minutes * 60)

        # Cleaning the data. THIS IS FOR OUR ELAAD NL DATASET
        # PLEASE CHANGE IF NEW DATA IS USED
        if config['data_collector'] == 'ELaadNL':
            processed_data = slotted_data[slotted_data.Energy_required != 0]
            processed_data['Departure_time'] = processed_data['Start_time'] + processed_data['Connected_time']
            processed_data = processed_data[processed_data.Departure_time <= processed_data.Start_time + 24]
            weirdline = (processed_data.Connected_time <= 18.01) & (processed_data.Connected_time > 17.99)
            processed_data = processed_data[-weirdline]
            processed_data = processed_data.reset_index()

            if config['verbose'] > 1:
                print(' ------------------- Created slotted data: Data cleaning - ')
                print(' \t\t Energy required > 0')
                print(' \t\t Departure time < Start time + 24 hours')
                print(' \t\t Removed the weird line in data')
        # save slotted data file
        processed_data.to_csv(os.path.join(save_location,"Slot_"+str(slot_minutes)+"_min_trans_data.csv") , index=False)

        return save_file_name,save_location