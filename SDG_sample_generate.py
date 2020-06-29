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
from modeling.generate_sample import run_script as generate_sample

model_folder = 'res/modeling/models'

# we load the latest model that was saved
save_loc = None             # for costom save name
save_name = None            # for costom save loc
process = 'PP'
lambda_mod = 'poisson_fit'

# we will pull the latest file with these models
list_of_files = glob.glob('res/modeling/models/SDG Model ('+str(process)+','+str(lambda_mod)+')*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
filename = os.path.join(latest_file)

with open(filename, 'rb') as f:
    x = pickle.load(f)
# these are the three saved models.
AM,MMc,MMe = x[0],x[1],x[2]
# here we can specify the horizon for data generation
horizon_start= datetime.datetime(2015,1 ,1)
horizon_end= datetime.datetime(2015,12,31)
# this function will generate the data using models AM,MMc and MMe in the given horizon.
gen_sample = generate_sample(AM=AM,MMc=MMc, MMe = MMe,
                horizon_start=horizon_start,horizon_end=horizon_end)

# the generated data will be saved in /res/generated samples
# we can specify save_loc and save_name to save the generated sample on a different location.
save_loc = save_loc if save_loc is not None else "res/generated samples"
save_name = save_name if save_name is not None else 'Generated sample ('+str(process)+','+str(lambda_mod)+') Horizon =' + str(horizon_start.date()) + "-to-" + str(
    horizon_end.date()) + '.csv'
gen_sample.to_csv(os.path.join(save_loc, save_name), index=False)

