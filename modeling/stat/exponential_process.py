# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# Class for exponential process
#
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
from modeling.stat.models import best_poly_fit
from modeling.stat.models import loess_fit
from handles.data_hand import get_slotted_data
from sklearn.linear_model import LinearRegression
from scipy.stats import kstest
from scipy.stats import expon
from scipy.stats import ttest_ind
import random
random.seed( 30 )


class exponential_process:

    def __init__(self,events,x,slotmin=60,x_meta=None,combine=None,variablity_lambda=True,log = True,normalize = True):
        # x is the numeric features lambda depends on.
        # x_meta are catagorical features that lambda depends on
        # Sesonality is when to loop the ts back. i.e. like 24 hours
        # x can be any factor levels. with _ in between each category. however, each catogory
        # should be defined by a numeric indicator
        self.x_names = np.array( x.columns )
        self.ts = np.array(events)
        self.x = np.array(x)
        self.x_meta=x_meta
        self.slotmin =  slotmin
        self.processed_data = self.get_combined_ts_data(combine=combine)
        self.def_scale_multiplier()
        self._variablity_lambda = variablity_lambda
        self._log = log
        self._normal = normalize

    def combine_timeslots(self,x,combine):
        p = x.copy()
        p[np.in1d(x, combine)] = combine[0]
        return p

    def poles_fun(self,d):
        return pd.DataFrame(d).apply(lambda x: 1/(x**3))

    def def_scale_multiplier(self):
        # this is based on emperical data
        average_mat = pd.DataFrame({'2014':[0.237053898,0.23033784,0.22646637,0.224855127,0.22145071,0.22017719,0.219680942],
        '2015':[0.190591233,0.185363899,0.183113651,0.180825924,0.179276851,0.179478113,0.17919847]}).T
        average_mat.columns = [1000,1100,1200,1300,1400,1500,1600]
        average_mat=average_mat.reset_index()
        average_mat=average_mat.melt(id_vars=["index"],var_name="Poles",value_name="Value")
        cols = ['year','poles','scale']
        average_mat.columns = cols
        average_mat[cols] = average_mat[cols].apply(pd.to_numeric, errors='coerce')
        average_mat['poles']=self.poles_fun(average_mat['poles'])
        regressor = LinearRegression()
        regressor.fit(average_mat[['year','poles']], average_mat['scale'])
        self.scale_multiplier_predictor = regressor
        self.reset_scale_multiplier()

    def reset_scale_multiplier(self):
        self._scale_multiplier = 1

    def avg_scale_pred(self,year,poles):
        return self.scale_multiplier_predictor.predict(np.array([year,
                                                              np.array(self.poles_fun([poles]))]).reshape(1, -1))

    def get_processed_data(self):
        diff_col_name = 'Aarrival_diff'
        delta_t = np.diff(self.ts, n=1).reshape(-1, 1)
        fin_d = pd.DataFrame(np.concatenate((delta_t, self.x[:-1, :]), axis=1))
        fin_d.columns = np.concatenate(
            (np.array(diff_col_name).reshape(-1, 1), np.array(self.x_names).reshape(-1, 1)), axis=0).flatten()
        fin_d[diff_col_name] = pd.to_numeric(fin_d[diff_col_name])
        # split the values in the factor that was provided to us
        split = fin_d[self.x_names[0]].str.split("_", -1)
        n = []
        for i in range(0, len(split[0])):
            fin_d['f' + str(i)] = split.str.get(i)#.astype(float)       # update this if code breaks
            n.append('f' + str(i))
        n.append(self.x_names[1])
        self.all_names = n
        fin_d = fin_d.sort_values(by=n)
        return fin_d

    def get_combined_ts_data(self,combine):
        # combine timeslots
        # if given argument = combine -- array of time slots to combine. we will replace these with
        # the first element of the combine array
        # start time internal is the timeslots to model the data on
        self.processed_data = self.get_processed_data()
        self.combine = combine
        if combine is None:
            self.combined_slots = False
            combined_timeslots = self.processed_data[self.x_names[1]]
        else:
            self.combined_slots = True
            combined_timeslots = self.combine_timeslots(self.processed_data[self.x_names[1]], combine=combine)
        self.processed_data['Start_time_internal'] = combined_timeslots
        return self.processed_data

    def get_slotted_data(self,data, slot_secs):
        return get_slotted_data(data=data,slot_secs=slot_secs)

    # ------------------------------------------- FITTING --------------------------------------------------------------

    def daywise_training_data(self,d,combine,fac1,fac2,f1,days,orignal_start_slot):
        # fac2 is out internal slots that are combined
        # it is also worth noting that we calculate the average for combined slots and then put them for
        # all the slots for that given duration
        if self.combined_slots:
            x = fac2[(fac1 == f1)]
            day = days[(fac1 == f1)]
            model_d = []
            for day_i in np.unique(day):

                model_d_temp = []
                for t_i in np.unique(x):
                    try:
                        model_d_temp.append([[t_i, expon.fit(pd.to_numeric(d[(x == t_i) & (day == day_i)]))[1], day_i]])
                    except:
                        continue
                model_d_temp = np.vstack(model_d_temp)
                scale_val = model_d_temp[(model_d_temp[:, 0] == combine[0])].flatten()[1]
                add = [[i, scale_val, day_i] for i in combine[1:]]
                model_d_temp = np.concatenate((model_d_temp, add))
                model_d.append(model_d_temp)
            model_d = np.vstack(model_d)
        else:
            x = orignal_start_slot[(fac1 == f1)]
            day = days[(fac1 == f1)]
            model_d = []
            for day_i in np.unique(day):
                model_d_temp = []
                for t_i in np.unique(x):
                    try:
                        model_d_temp.append([[t_i, expon.fit(pd.to_numeric(d[(x == t_i) & (day == day_i)]))[1], day_i]])
                    except:
                        continue
                model_d_temp = np.vstack(model_d_temp)
                model_d.append(model_d_temp)
            model_d = np.vstack(model_d)
        return model_d

    def mean_model(self,data,x,data_save,x_save):

        ks_t_D = pd.DataFrame()
        ks_t_pval = pd.DataFrame()
        t_t_pval = pd.DataFrame()
        exp_loc = pd.DataFrame()
        exp_scale = pd.DataFrame()
        time_slot = pd.DataFrame()

        for f2 in np.unique(x):
            d = pd.to_numeric(np.array(data[(x==f2)]))
            loc, scale = expon.fit(d)
            # ks test
            D , kspval = kstest(d,'expon')
            # ttest  - one sided
            sample2 = np.random.exponential(scale, size=d.shape[0])
            val , pval = ttest_ind(d,sample2)

            # if we have combined data then add same model to all combined timeslots
            if self.combined_slots and f2 == self.combine[0]:
                for var in self.combine:
                    exp_loc = exp_loc.append(pd.DataFrame([loc]))
                    exp_scale = exp_scale.append(pd.DataFrame([scale]))
                    ks_t_D = ks_t_D.append(pd.DataFrame([D]))
                    ks_t_pval = ks_t_pval.append(pd.DataFrame([kspval]))
                    t_t_pval = t_t_pval.append(pd.DataFrame([pval / 2]))
                    # add timeslot
                    time_slot = time_slot.append([var])

            else:
                exp_loc = exp_loc.append(pd.DataFrame([loc]))
                exp_scale = exp_scale.append(pd.DataFrame([scale]))
                ks_t_D = ks_t_D.append(pd.DataFrame([D]))
                ks_t_pval = ks_t_pval.append(pd.DataFrame([kspval]))
                t_t_pval = t_t_pval.append(pd.DataFrame([pval / 2]))
                # add timeslot
                time_slot = time_slot.append([f2])


        # this is the final fit
        fit = pd.DataFrame()
        fit[[self.x_names[1]]] = time_slot
        fit['Exp_loc'] = np.array(exp_loc).flatten()
        fit['Exp_scale'] = np.array(exp_scale).flatten()
        fit['KS_D'] = np.array(ks_t_D).flatten()
        fit['KS_PVal'] = np.array(ks_t_pval).flatten()
        fit['Ttest_PVal'] = np.array(t_t_pval).flatten()

        # if self._log:
        #     data_save = np.log(data_save)
        # if self._normal:
        #     day_max = pd.DataFrame({'time':x,'scale':data,'day':days} )
        #     day_max = day_max.groupby("day")["scale"].transform(max)
        #     data = data/day_max
        #     scalings = np.unique(day_max)
        # else:
        #     scalings = 1


        return fit,data_save,x_save

    def poly_model(self, model_d,verbose=1):
        # remove if the data is zero somewhere
        index = np.where(model_d[:,1] == 0)[0]
        data = np.delete(model_d[:,1],index)
        x = np.delete(model_d[:,0],index)
        days = np.delete(model_d[:,2],index)
        # we do not fit scale in poly model. we fit 1/scale i.e lambda
        if self.inverse:
            data = 1/data
        if self._log:
            data = np.log(data)
        if self._normal:
            day_max = pd.DataFrame({'time':x,'scale':data,'day':days} )
            day_max = day_max.groupby("day")["scale"].transform(max)
            data = data/day_max
            scalings = np.unique(day_max)
        else:
            scalings = 1

        poly_fit = best_poly_fit(y=np.array(data), x=np.array(x), max_deg=self.max_poly_deg,scoring=self._scoring,verbose=verbose)
        return poly_fit,data,x,scalings

    def loess_model(self, model_d, poly_degree=5):
        # remove if the data is zero somewhere
        index = np.where(model_d[:, 1] == 0)[0]
        data = np.delete(model_d[:, 1], index)
        x = np.delete(model_d[:, 0], index)
        days = np.delete(model_d[:, 2], index)
        # we do not fit scale in poly model. we fit 1/scale i.e lambda
        if self.inverse:
            data = 1/data
        if self._log:
            data = np.log(data)
        if self._normal:
            day_max = pd.DataFrame({'time':x,'scale':data,'day':days} )
            day_max = day_max.groupby("day")["scale"].transform(max)
            data = data/day_max
            scalings = np.unique(day_max)
        else:
            scalings = 1
        loess_fit_n = loess_fit(yvals=np.array(data), xvals =np.array(x), poly_degree=self.poly_deg,alpha=self.alpha)
        self._loess_deg = poly_degree
        return loess_fit_n,data,x,scalings

    def fit(self,lambda_mod='poly',combine=np.arange(1,7),inverse=True,poly_deg=1,max_poly_deg = 5,alpha=0.2,
            year=2015,poles=1677,verbose=1):

        # ------------------------------ Prepration -------------------------------------------------
        # create dataset for modeling
        # if continous = True, we replace the values of means with continous values
        self.lambda_mod = lambda_mod
        # used for poly regression
        self.max_poly_deg = max_poly_deg
        self._scoring = 'my_scorer'
        # used for loess
        self.poly_deg = poly_deg
        self.alpha = alpha
        # inverse is because we fit 1/scale which means it is lambda and not scale
        self.inverse = inverse
        # create the scale multipler for this function
        self._fit_year = int(year)
        self._fit_poles = int(poles)
        self._scale_multiplier_fit = self.avg_scale_pred(year=self._fit_year,poles=self._fit_poles)
        # this is done because poly model has -ve starting
        self._scale_old = pd.Series(self._scale_multiplier_fit)

        # pull the processed dataset after passingon the combine argument
        fin_d = self.get_combined_ts_data(combine=combine)

        # ------------------------------ Modeling -------------------------------------------------
        model_data = fin_d['Aarrival_diff'].copy()
        days = fin_d[self.x_names[2]].copy()
        self.ts_diff = model_data
        fac1 = fin_d[self.x_names[0]]
        fac2 = fin_d['Start_time_internal'] # usually timeslot
        orignal_start_slot = fin_d[self.x_names[1]]
        fit = []
        # model for mean values in each slot and fac

        for f1 in np.unique(fac1):
            if verbose > 1: print(' \t\t Fitting parameters for factor : ', str(f1))
            d = model_data[(fac1 == f1)]
            # fac2 is out internal slots that are combined
            # it is also worth noting that we calculate the average for combined slots and then put them for
            # all the slots for that given duration
            model_d = self.daywise_training_data(d, combine, fac1, fac2, f1, days, orignal_start_slot)

            if lambda_mod == 'mean':
                if self.combined_slots:
                    x = fac2[(fac1 == f1)]
                else:
                    x = orignal_start_slot[(fac1 == f1)]
                temp_fit = self.mean_model(data=d, x=x,data_save=model_d[:,1],x_save= model_d[:, 0])

            # model for poly values in each slot and fac
            if lambda_mod == 'poly':
                temp_fit = self.poly_model(model_d=model_d,verbose=verbose)

            # model for gpr values in each slot and fac
            if lambda_mod == 'gpr':
                temp_fit = self.gpr_model(data=model_d[:, 1], x=model_d[:, 0])

            # model for loess values in each slot and fac
            if lambda_mod == 'loess':
                # inverse is because we fit 1/scale which means it is lambda and not scale
                # Model
                #   - [0 - 1 - 2] loess model, y , x
                #       - [0][0 - 1] regsDF and evalDF
                #           -[0][1][ v - b ] fitted x and y
                temp_fit = self.loess_model(model_d=model_d)

            fit.append([f1, temp_fit])
            # names to be added to final fitted variable
            names = np.append(self.x_names[0], 'Model')

        fit_merged = [list(x) for x in zip(*fit)]
        fit_named = dict(zip(names, fit_merged))
        self._best_fit = fit_named
        return fit_named

    # ------------------------------------------- PREDICT --------------------------------------------------------------

    def pop_model(self, f1):
        # this function returns the poly model for the given f1
        # this method can be called for both poly and loess models
        # returns a tuple, that means we have to select the first element to get the model
        # tuple has model, y, and x values
        selector_boolan = ((np.array(self._best_fit[self.x_names[0]]) == f1))
        selector = np.where(selector_boolan.flatten())[0]
        Model = self._best_fit['Model']
        M_topop = Model[int(selector)]
        return M_topop

    def treat_scale(self, scale):
        if scale.empty:
            scale = self._scale_old
        else:
            if np.array(scale)[0] <= 0 or np.array(scale)[0] > 1:
                scale = self._scale_old
                # save l if needed
        self._scale_old = scale.copy()
        return scale * self._scale_multiplier

    def predict_mean_mod(self, f1, f2, val='Exp_scale', return_raw=False):
        if return_raw:
            model = self.pop_model(f1=f1)[0]
            Model_t = model[(model[self.x_names[1]] == f2)]
            scale = Model_t[val]
            return scale

        # this function predicts the values
        if self._todays_random is None:
            model = self.pop_model(f1=f1)
            n_days = (len((model[2])) + 1) / 24
            num = random.choice(np.arange(0, n_days))
            self._todays_random = np.arange(24 * num, (24 * (num + 1))) - 1

        if self._variablity_lambda and val == 'Exp_scale':
            model = self.pop_model(f1=f1)
            index = np.where(model[2] == f2)
            try:
                common = np.intersect1d(index, self._todays_random)
                scale = pd.Series(model[1][int(common[0])])
            except:
                Model_t = model[0][(model[0][self.x_names[1]] == f2)]
                scale = Model_t[val]
        else:
            model = self.pop_model(f1=f1)[0]
            Model_t = model[(model[self.x_names[1]] == f2)]
            scale = Model_t[val]
        if val != 'Exp_scale':
            return scale
        else:
            return self.treat_scale(scale=scale)

    def predict_poly_mod(self, f1, time, return_raw=False):
        # this function predicts the values
        model = self.pop_model(f1=f1)
        prediction = model[0].predict(time.reshape(-1, 1)).flatten()
        if return_raw:
            return prediction
        std = model[0]._total_std

        if self._todays_random is None:
            self._todays_random = random.choice(model[3])

        if self._variablity_lambda:
            # prediction = random_t_dist.rvs(df=1, loc=prediction, scale=std, size=1)
            prediction = prediction * self._todays_random

        if self._log:
            prediction = np.exp(prediction)
        if self.inverse:
            prediction = 1 / prediction
        scale = pd.Series(prediction)
        scale = float(self.treat_scale(scale=scale))

        return float(scale), std

    def predict_loess_mod(self, f1, time, return_raw=False):
        # this function predicts the values
        # Model
        #   - [0 - 1 - 2] loess model, y , x
        #       - [0][0 - 1] regsDF and evalDF
        #           -[0][1][ v - b ] fitted x and y
        model = self.pop_model(f1=f1)
        index_in_eval = np.digitize([time], model[0][1]['v']).flatten()
        coff = np.array(model[0][1]['b'][index_in_eval].iloc[0]).astype(float)
        std = np.array(model[0][1]['std'][index_in_eval].iloc[0]).astype(float)
        x1 = np.array([time] * (1 + self.poly_deg)).flatten()
        x2 = np.arange(self.poly_deg + 1).flatten()
        X = np.power(x1, x2)
        prediction = sum(coff * X)
        if return_raw:
            return prediction
        # = model.predict(time.reshape(-1, 1)).flatten()
        # this will be generated once a day
        if self._todays_random is None:
            self._todays_random = random.choice(model[3])

        if self._variablity_lambda:
            # prediction = random_t_dist.rvs(df=1, loc=prediction, scale=std, size=1)
            prediction = prediction * self._todays_random

        if self._log:
            prediction = np.exp(prediction)
        if self.inverse:
            prediction = 1 / prediction
        scale = pd.Series(prediction)
        scale = float(self.treat_scale(scale=scale))

        return float(scale), std

    def predict_day(self, X_test, Start_time, slot,
                    year=None, poles=None, variablity_lambda=False,verbose=0):

        # here we generate a days time series
        if verbose > 2: print(' \t\t Generating arrivals using fitted model for : ', str(np.array( X_test)[0][0]))
        t_now = Start_time
        ts = []

        # update X_test if we are predicting for any other year
        X_test = X_test.replace({str(int(year)): str(self._fit_year)}, regex=True) if year is not None else X_test

        # generate scale multiplier
        y = year if year is not None else self._fit_year
        p = poles if poles is not None else self._fit_poles
        self._scale_multiplier = self.avg_scale_pred(year=y, poles=p) / self._scale_multiplier_fit
        self._scale_min = float(1000)
        self._variablity_lambda = variablity_lambda
        self._todays_random = None

        while t_now <= 24.00:
            ts.append(float(t_now))
            t_now_slot = int(self.get_slotted_data(data=t_now, slot_secs=slot))
            if self.lambda_mod == 'mean':
                scale = self.predict_mean_mod(f1=np.array(X_test), f2=t_now_slot)

            if self.lambda_mod == 'poly':
                scale, std = self.predict_poly_mod(f1=np.array(X_test), time=t_now)

            if self.lambda_mod == 'loess':
                scale, std = self.predict_loess_mod(f1=np.array(X_test), time=t_now)

            # update minimum scale
            if self._scale_min > float(scale):
                self._scale_min = float(scale)

            t_diff = np.random.exponential(scale, size=1)
            # update t_now
            t_now = t_now + t_diff

        return np.array(ts), t_now, self._scale_min
