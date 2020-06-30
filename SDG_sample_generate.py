# ---------------------------------------------- ML 20/04/2020 -----------------------------------------------------#
#
# Generate a sample of EV sessions data.
#               This file can be used to generate the sample of a data using the saved SDG model.
#                   - We pick the latest saved model
#                   - Each SDG model is a list of 3 models - AM,MMc and MMe
#                       Arrival model, mixture model for connected time and mixture model for energy required
#                   - Sample is generated for the given horizon.
#
#  dependencies: modeling.generate_sample.py

# -------------------------------------------------------------------------------------------------------------------- #

import glob
import os
import pickle
import datetime
import argparse
# load the config file
import json
config = json.load(open('config.json'))

def main(args):
    # here we can specify the horizon for data generation
    os.makedirs(config['dir_names'][config['dir_names']['generated_samples_name']], exist_ok=True)
    try:
        horizon_start = datetime.datetime.strptime(str(args['start_date']), '%d/%m/%Y')
        horizon_end = datetime.datetime.strptime(str(args['end_date']), '%d/%m/%Y')
        if config['verbose'] > 0: print(' ------------------- Generating data -------------------')
        if config['verbose'] > 0: print(' \t\t Horizon : '+str(horizon_start) + ' to '+ str(horizon_end))
    except:
        print(" Please provide starting and ending date for data generation!")
        return

    # we will pull the SDG pickle file
    try:
        if str(args['use']) == 'default':
            model_loc = config['models']['loc']
            model_name = config['models'][str(args['model']+args['lambdamod'])]
            file = os.path.join(model_loc,model_name)
        if str(args['use']) == 'latest':
            list_of_files = glob.glob(config['dir_names']['models_folder_name']+ '/SDG Model (' + str(args['model']) + ',' + str(
                args['lambdamod']) + ')*')  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            file = os.path.join(latest_file)
        if config['verbose'] > 0: print(' \t\t Using : ' + str(args['use']) + ' model ')
        if config['verbose'] > 0: print(' \t\t location : ' + str(file))
    except:
        print(" \t\t Please select a trained SDG model!\n"
              " \t\t argument '-use default' can be used to use default models\n"
              " \t\t and '-use latest' for using the latest trained models ")
        return

    with open(file, 'rb') as f:
        x = pickle.load(f)
    print(x)
    # these are the three saved models.
    AM,MMc,MMe = x[0],x[1],x[2]

    # we load the latest model that was saved
    save_loc = config['dir_names']['generated_samples_name']
    save_name = 'Generated sample ('+str(args['model'])+','+str(args['lambdamod'])+') Horizon =' + str(horizon_start.date()) + "-to-" + str(
        horizon_end.date()) + '.csv'


    # this function will generate the data using models AM,MMc and MMe in the given horizon.
    from modeling.generate_sample import generate_sample
    gen_sample = generate_sample(AM=AM,MMc=MMc, MMe = MMe,
                    horizon_start=horizon_start,horizon_end=horizon_end)
    gen_sample.to_csv(os.path.join(save_loc, save_name), index=False)

    print(" EV sessions data saved at : ", save_loc)
    print(" EV sessions data filename : ", save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to SDG sample generate: \n '
                                                 'Generates a sample of EV sessions data')
    parser.add_argument('-start_date',
                        help='first date of the horizon for data generation \n'
                             'format: dd/mm/YYYY')
    parser.add_argument('-end_date',
                        help='last date of the horizon for data generation \n'
                             'format: dd/mm/YYYY')
    parser.add_argument('-use', default='default',
                        help='which kind of models to use.  \n'
                             '\t\t "default" for using the default models \n'
                             '\t\t "latest" for using the lastest trained models')
    parser.add_argument('-model', default='AC',
                        help='modeling method to be used for modeling arrival times: \n'
                             '\t\t AC for arrival count models \n'
                             '\t\t IAT for inter-arrival time models')
    parser.add_argument('-lambdamod', default='poisson_fit',
                        help='Method to be used for modeling lambda:\n'
                             '\t\t AC: has two options, poisson_fit/neg_bio_reg \n'
                             '\t\t IAT: has three options, mean/loess/poly')
    parser.add_argument('-verbose', default=3,
                        help='0 to print nothing; >0 values for printing more information. Possible values:0,1,2,3 (integer)')

    args = parser.parse_args()
    main(vars(args))