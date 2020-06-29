# -------------------------------------------------- ML 28/11/2019 ----------------------------------------------------#
#
# used to operate on SDG final model.
#
# -------------------------------------------------------------------------------------------------------------------- #


from modeling.stat.exponential_process import exponential_process
from modeling.stat.poisson_process import poisson_process
import numpy as np

def remove_raw_data(model):
    # here we remove the raw data that is saved in the models. usually in the _best_fit function/which
    # is responsible for the models.
    if isinstance(model, exponential_process) or isinstance(model, poisson_process):
        for i in range(0,len(model._best_fit['Model'])):
            lst = list(model._best_fit['Model'][int(i)])
            lst[1] = [0]
            lst[2] = [0]
            if model.lambda_mod == 'loess':
                lst1 = list(lst[0])
                lst1[0] = [0]
                lst[0] = tuple(lst1)
            model._best_fit['Model'][int(i)] = tuple(lst)


        model.ts = 'Data Deleted'
        model.ts_diff = 'Data Deleted'
        model.x = 'Data Deleted'
        model.processed_data = 'Data Deleted'
    else:
        model.y = 'Data Deleted'
        model.initilizations = 'Data Deleted'
        model.x = 'Data Deleted'
        model.processed_data = 'Data Deleted'

    return model

def get_model_log(model):
    # Here we create a log/data file that will be saved as csv for the user
    if isinstance(model, exponential_process) or isinstance(model, poisson_process):
        model_log = {}
        model_log['Arrival model'] = "###################################################################"
        model_log['Process'] = model.__class__.__name__
        model_log['lambda_mod'] = model.lambda_mod
        model_log['Year'] = model._fit_year
        model_log['Randomization'] = model._variablity_lambda
        model_log['Combined_timeslots'] = model.combine
        model_log["MODELS "] = "-----------------------------------"
        for i in np.arange(0, len(model._best_fit['Factor'])):
            model_log[str(model._best_fit['Factor'][i])] = \
                model._best_fit['Model'][i][0]
    else:
        model_log = {}
        model_log['Mixture model'] = "###################################################################"
        model_log['For'] = model.model_for
        model_log['Mixture_of'] = model._mix
        model_log['Optimizarion'] = model._method
        model_log['Combined_timeslots'] = model.combine
        model_log["MODELS "] = "-----------------------------------"
        for i in np.arange(0, len(model._best_fit['Factor'])):
            model_log[str(model._best_fit['Factor'][i]) + " " + str(model._best_fit['Start_time_slot'][i])] = \
            model._best_fit['Model'][i]
    return model_log
