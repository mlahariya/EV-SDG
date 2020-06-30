# ---------------------------------------------- ML 25/10/2019 -----------------------------------------------------#
#
# We just create a resources folder here. Which will contain all our data and start the preprocessing here.
# All Scripts are present in preprocessing folder. We just call two scripts from here.
#   - Trans_analysis: Analysis of each transaction, this will generate a processed 15 min trans data file in resources
#     csv produced includes the clusters, information about arrivals, daily/weekly distribution, and other properties
#
#   - ts_analysis: Analysis of data in form of time series, this will create a TimeSeries 15 minute data file
#     csv produced includes the arrivals/departure properties, along with information about each slot.
#
# IMP: Processed Transaction file is created for one year. That year has to be mentioned here.
#      However, time series data is created for all years.
#    : Please make sure that transactions file (real world raw data file) is present on root directory and
#      has the following columns
#       Column name     | Description                               | Data format
#       Started         | Starting date and time of the EV session  | datetime (dd/mm/YYY HH:MM:SS)
#       ConnectedTime   | Connection time of the EV session         | Hours (float)
#       TotalEnergy     | Requested energy of the EV session        | kWh (float)
#       ChargePoint     | Charging station                          | Categorical (str)
#
# -------------------------------------------------------------------------------------------------------------------- #

# import libraries
import argparse
import json
import warnings
warnings.filterwarnings("ignore")


def main(args):
    config = json.load(open('config.json'))
    config['Year'] = int(args['Year'])
    config['slot_mins'] = int(args['Slotmins'])
    config['verbose'] = int(args['verbose'])
    config['create_plots'] = bool(args['create_plots'])
    config['transactions_filename'] = str(args['Sessions_filename'])
    config['data_collector'] = 'ELaadNL'
    json.dump(config, open('config.json', 'w'))

    # create directories
    from dir_gen import dir_create
    val = dir_create(folder_name=str(args['res_folder']))
    if not(val) :
        print("Raw Data file not found at :" , config['res_folder'])
        return

    # generate processed data
    from preprocess.prepare_clustered_data import create_processed_data
    create_processed_data()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to preprocess data: \n '
                                                 'Creates session and pole clusters and time series data for modeling')
    parser.add_argument('-Year', default=2015,
                        help='Year for modeling (integer)')
    parser.add_argument('-Slotmins', default=60,
                        help='Minutes in each timeslot (integer)')
    parser.add_argument('-create_plots', default=True,
                        help='indicator for creating plots')
    parser.add_argument('-Sessions_filename', default='transactions.csv',
                        help='Name of the file contaning raw data. This file must be present in /res folder (str)')
    parser.add_argument('-res_folder', default='res',
                        help='Locaiton for raw data file. default is "/res" inside this directory'
                             'EV sessions files must be present here (string)')
    parser.add_argument('-verbose', default=1,
                        help='0 to print nothing; >0 values for printing more information. '
                             'Possible values:0,1,2,3 (integer)')

    args = parser.parse_args()
    main(vars(args))


