# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# This is the class for poisson process
#
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import math
from handles.data_hand import get_slotted_data
from sklearn.linear_model import LinearRegression
from scipy.stats import kstest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from modeling.stat.models import fit_neg_binom
from scipy.stats import expon,gamma,nbinom
import random
random.seed( 30 )


class poisson_process:

    def __init__(self,events,x,slotmin=60,sesonality=24.00,x_meta=None,combine=None,variablity_lambda=True):
        # x is the numeric features lambda depends on.
        # x_meta are catagorical features that lambda depends on
        # Sesonality is when to loop the ts back. i.e. like 24 hours
        # x can be any factor levels. with _ in between each category. however, each catogory
        # should be defined by a numeric indicator
        self.x_names = np.array( x.columns )
        self.ts = np.array(events)
        self.x = np.array(x)
        self.x_meta=x_meta
        self.slotmin = slotmin
        self.sesonality = float( sesonality )
        self.processed_data = self.get_combined_ts_data(combine=combine)
        self.def_scale_multiplier()
        self._variablity_lambda = variablity_lambda

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

    def discreet_fit_model(self,data,x):

        data_gamma = pd.DataFrame({'days':data, 'arrivalslot':x,'indicator':1})
        data_gamma = data_gamma.groupby(['days','arrivalslot']).agg(['count']).reset_index()
        data_gamma.columns = ['days', 'arrivalslot','count']

        data_save = data_gamma['count']
        x_save =  data_gamma['arrivalslot']


        ks_t_D = pd.DataFrame()
        ks_t_pval = pd.DataFrame()
        t_t_pval = pd.DataFrame()
        exp_loc = pd.DataFrame()
        exp_scale = pd.DataFrame()
        exp_shape = pd.DataFrame()
        time_slot = pd.DataFrame()
        pos_l = pd.DataFrame()
        neg_bio_r = pd.DataFrame()
        neg_bio_p = pd.DataFrame()

        for f2 in np.unique(data_gamma['arrivalslot']):
            d = pd.to_numeric( data_gamma[data_gamma['arrivalslot'] == f2]['count'] )
            # poission
            lam = np.mean(d)
            # gamma
            alpha,loc, beta = gamma.fit(d,loc=0)
            # ks test
            D , kspval = kstest(d,'gamma', args=(alpha,loc,beta))
            # ttest  - one sided
            # sample2 = gamma.rvs(a = alpha, loc=loc, scale=beta, size=d.shape[0])
            val , pval = 0,0 #ttest_ind(d,sample2)
            # neg_binom
            r,p = fit_neg_binom(vec=np.array(d).flatten(),init=0.0000001)



            # if we have combined data then add same model to all combined timeslots
            if self.combined_slots and f2 == self.combine[0]:
                for var in self.combine:
                    pos_l = pos_l.append(pd.DataFrame([lam]))
                    exp_loc = exp_loc.append(pd.DataFrame([loc]))
                    exp_shape = exp_shape.append(pd.DataFrame([alpha]))
                    exp_scale = exp_scale.append(pd.DataFrame([beta]))
                    neg_bio_r = neg_bio_r.append(pd.DataFrame([r]))
                    neg_bio_p = neg_bio_p.append(pd.DataFrame([p]))


                    ks_t_D = ks_t_D.append(pd.DataFrame([D]))
                    ks_t_pval = ks_t_pval.append(pd.DataFrame([kspval]))
                    t_t_pval = t_t_pval.append(pd.DataFrame([pval / 2]))
                    # add timeslot
                    time_slot = time_slot.append([var])

            else:
                pos_l = pos_l.append(pd.DataFrame([lam]))
                exp_loc = exp_loc.append(pd.DataFrame([loc]))
                exp_shape = exp_shape.append(pd.DataFrame([alpha]))
                exp_scale = exp_scale.append(pd.DataFrame([beta]))
                neg_bio_r = neg_bio_r.append(pd.DataFrame([r]))
                neg_bio_p = neg_bio_p.append(pd.DataFrame([p]))

                ks_t_D = ks_t_D.append(pd.DataFrame([D]))
                ks_t_pval = ks_t_pval.append(pd.DataFrame([kspval]))
                t_t_pval = t_t_pval.append(pd.DataFrame([pval / 2]))
                # add timeslot
                time_slot = time_slot.append([f2])


        # this is the final fit
        fit = pd.DataFrame()
        fit[[self.x_names[1]]] = time_slot
        fit['gamma_loc'] = np.array(exp_loc).flatten()
        fit['gamma_scale'] = np.array(exp_scale).flatten()
        fit['gamma_shape'] = np.array(exp_shape).flatten()
        fit['KS_D'] = np.array(ks_t_D).flatten()
        fit['KS_PVal'] = np.array(ks_t_pval).flatten()
        fit['Ttest_PVal'] = np.array(t_t_pval).flatten()
        fit['Poisson_lam'] = np.array(pos_l).flatten()
        fit['Negbio_r'] = np.array(neg_bio_r).flatten()
        fit['Negbio_p'] = np.array(neg_bio_p).flatten()


        return fit,data_save,x_save

    def neg_bio_reg_fit_model(self,data,x):

        data_gamma = pd.DataFrame({'days': data, 'arrivalslot': x, 'indicator': 1})
        data_gamma = data_gamma.groupby(['days', 'arrivalslot']).agg(['count']).reset_index()
        data_gamma.columns = ['days', 'arrivalslot', 'count']

        data_save = data_gamma['count']
        x_save = data_gamma['arrivalslot']

        nb_mu = pd.DataFrame()
        nb_p = pd.DataFrame()
        nb_n = pd.DataFrame()
        nb_alpha = pd.DataFrame()
        time_slot = pd.DataFrame()
        # data_gamma.to_csv("aaaaaaaaaaaaaaaaaa.csv")
        for f2 in np.unique(data_gamma['arrivalslot']):
            d = pd.to_numeric(data_gamma[data_gamma['arrivalslot'] == f2]['count'])
            X_train = np.ones(len(d))
            try:
                df_train = pd.DataFrame({'counts':d,'Intercept':X_train})

                # Calculating alpha = shape parameter
                # theta = (1/alpha) = r = number of sucess
                # Using the statsmodels GLM class, train the Poisson regression model on the training data set
                poisson_training_results = sm.GLM(d, X_train, family=sm.families.Poisson()).fit()
                df_train['BB_LAMBDA'] = poisson_training_results.mu
                df_train['AUX_OLS_DEP'] = df_train.apply(
                    lambda x: ((x['counts'] - x['BB_LAMBDA']) ** 2 - x['counts']) / x['BB_LAMBDA'], axis=1)
                ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
                aux_olsr_results = smf.ols(ols_expr, df_train).fit()
                alpha = aux_olsr_results.params[0]
                # introducing a minimum liimit on alpha
                # -- putting alpha = 0.00001 trigggers poisson distribution
                if alpha <= 0:
                    alpha = 0.00001 # ---> this will trigger poisson while prediciton
                    # alpha = 0.25      # this just introductes a min limit on alpha

                # # use this when we dont want to use calculated alpha
                # alpha = 0.2

                # calculating the mean parameter mu
                nb2_training_results = sm.GLM(d.astype(float), X_train.astype(float),
                                              family=sm.families.NegativeBinomial(alpha = alpha)).fit()
                mean = float( np.exp(nb2_training_results.params) )# float(np.mean(d))
                # calculate n and p
                n = float(1/alpha)
                var = mean + 1 / n * mean ** 2
                p = float(1-((var - mean) / var))
                # var = mean + (np.power(mean,2)*alpha)
                # n = float((np.power(mean,2))/ (var - mean))
                # p = float((var - mean)/var)
            except:
                n,p,mean,alpha = -1,-1,-1,-1

            # if we have combined data then add same model to all combined timeslots
            if self.combined_slots and f2 == self.combine[0]:
                for var in self.combine:
                    nb_mu = nb_mu.append(pd.DataFrame([mean]))
                    nb_p = nb_p.append(pd.DataFrame([p]))
                    nb_n = nb_n.append(pd.DataFrame([n]))
                    nb_alpha = nb_alpha.append(pd.DataFrame([alpha]))
                    time_slot = time_slot.append([var])

            else:
                nb_mu = nb_mu.append(pd.DataFrame([mean]))
                nb_p = nb_p.append(pd.DataFrame([p]))
                nb_n = nb_n.append(pd.DataFrame([n]))
                nb_alpha = nb_alpha.append(pd.DataFrame([alpha]))
                # add timeslot
                time_slot = time_slot.append([f2])

            # this is the final fit
        fit = pd.DataFrame()
        fit[[self.x_names[1]]] = time_slot
        fit['nb_n'] = np.array(nb_n).flatten()
        fit['nb_p'] = np.array(nb_p).flatten()
        fit['nb_mu'] = np.array(nb_mu).flatten()
        fit['nb_alpha'] = np.array(nb_alpha).flatten()

        return fit,data_save,x_save

    def fit(self,lambda_mod='expon_fit',combine=np.arange(1,7),year=2015,poles=1677,verbose=1):

        # ------------------------------ Prepration -------------------------------------------------
        # create dataset for modeling
        # if continous = True, we replace the values of means with continous values
        self.lambda_mod = lambda_mod
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
        for f1 in np.unique(fac1):
            if verbose > 1: print(' \t\t Fitting parameters for factor : ', str(f1))
            d = model_data[(fac1 == f1)]
            # fac2 is out internal slots that are combined
            # it is also worth noting that we calculate the average for combined slots and then put them for
            # all the slots for that given duration
            model_d = self.daywise_training_data(d, combine, fac1, fac2, f1, days, orignal_start_slot)

            if self.lambda_mod == 'poisson_fit':
                if self.combined_slots:
                    x = fac2[(fac1 == f1)]
                else:
                    x = orignal_start_slot[(fac1 == f1)]
                alldays = days[(fac1 == f1)]
                temp_fit = self.discreet_fit_model(data=alldays, x=x)

            if self.lambda_mod == 'neg_bio_reg':
                # temp_fit = neg_bio_reg_fit
                # we save n p mu and alpha for neg binomial
                if self.combined_slots:
                    x = fac2[(fac1 == f1)]
                else:
                    x = orignal_start_slot[(fac1 == f1)]
                alldays = days[(fac1 == f1)]
                temp_fit = self.neg_bio_reg_fit_model(data=alldays, x=x)

            fit.append([f1, temp_fit])
            # names to be added to final fitted variable
            names = np.append(self.x_names[0], 'Model')

        fit_merged = [list(x) for x in zip(*fit)]
        fit_named = dict(zip(names, fit_merged))
        self._best_fit = fit_named
        return fit_named

    # ------------------------------------------- PREDICT --------------------------------------------------------------

    def pop_model(self,f1):
        # this function returns the poly model for the given f1
        # this method can be called for both poly and loess models
        # returns a tuple, that means we have to select the first element to get the model
        # tuple has model, y, and x values
        selector_boolan = ((np.array(self._best_fit[self.x_names[0]]) == f1))
        selector = np.where(selector_boolan.flatten())[0]
        Model = self._best_fit['Model']
        M_topop = Model[int(selector)]
        return M_topop

    def predict_poission_fit_mod(self,f1,f2,val='Poisson_lam',return_raw = False):

        if return_raw:
            model = self.pop_model(f1=f1)[0]
            Model_t = model[(model[self.x_names[1]] == f2)]
            lam = Model_t['Poisson_lam']
            return lam




        if self._variablity_lambda and val =='Poisson_lam':
            # this function predicts the values
            if self._todays_random is None:
                model = self.pop_model(f1=f1)
                n_days = (len((model[2])) + 1) / 24
                num = random.choice(np.arange(0, n_days))
                self._todays_random = np.arange(24 * num, (24 * (num + 1))) - 1


            model = self.pop_model(f1=f1)
            index = np.where(model[2] == f2)
            try:

                # update this when variability is decided
                # common = np.intersect1d(index, self._todays_random)
                # n_arr = int(model[1][int(common[0])])
                # shape = 'useabs'
                # loc = 0
                # scale = n_arr
                Model_t = model[0][(model[0][self.x_names[1]] == f2)]
                lam = Model_t['Poisson_lam']

            except:
                model = self.pop_model(f1=f1)[0]
                Model_t = model[(model[self.x_names[1]] == f2)]
                lam = Model_t['Poisson_lam']

        else:

            model = self.pop_model(f1=f1)[0]
            Model_t = model[(model[self.x_names[1]] == f2)]
            lam = Model_t[val]

        return lam

    def predict_neg_bio_reg_fit_mod(self,f1,f2):

        model = self.pop_model(f1=f1)[0]
        Model_t = model[(model[self.x_names[1]] == f2)]
        n = Model_t['nb_n']
        p = Model_t['nb_p']
        mu = Model_t['nb_mu']

        # df_split = pd.DataFrame({"fac1":[f1[0,0]]}, index=[0])
        # splited = df_split["fac1"].str.split("_", -1)
        # X = pd.DataFrame(np.repeat(0,24)).T
        # X.columns = np.arange(1,25)
        # X.iloc[:,f2-1] = 1
        # nb2_predictions = model.get_prediction(X.astype(float))
        # n_arr = float(nb2_predictions.predicted_mean)
        # n_arr = np.floor(n_arr)
        # print(X)
        # print(n_arr)
        return n,p,mu

    def predict_day(self, X_test, Start_time, slot,
                    year=None,poles=None,variablity_lambda=False):

        # here we generate a days time series
        print('Generating time series for :',X_test)
        print('Generating .... ')
        t_now = Start_time
        ts = []

        # update X_test if we are predicting for any other year
        X_test = X_test.replace({str(int(year)):str(self._fit_year)},regex=True) if year is not None else X_test

        # generate scale multiplier
        y = year if year is not None else self._fit_year
        p = poles if poles is not None else self._fit_poles
        self._scale_multiplier = self.avg_scale_pred(year=y,poles=p)/self._scale_multiplier_fit
        self._scale_min = float(1000)
        self._variablity_lambda = variablity_lambda
        self._todays_random = None

        while t_now <=24.00:

            t_now_slot = int(self.get_slotted_data(data=t_now,slot_secs= slot))
            if self.lambda_mod == 'poisson_fit':
                lam = self.predict_poission_fit_mod(f1=np.array(X_test), f2=t_now_slot)
                scale = lam.copy()
                if not(math.isnan(lam)):
                    n_arrivals = np.random.poisson(lam=lam,size = 1)

            if self.lambda_mod == 'neg_bio_reg':
                n,p,mu = self.predict_neg_bio_reg_fit_mod(f1=np.array(X_test), f2=t_now_slot)
                scale = n.copy()
                if float(n) > 0:
                    # check if the neg binom dist tends to poisson
                    if round(float(n)) != 100000:
                        n_arrivals = nbinom.rvs(n=n[0],p=p[0],size=1)
                    else:
                        n_arrivals = np.random.poisson(lam=mu[0],size = 1)

            # update minimum scale
            if self._scale_min > float(scale):
                self._scale_min = float(scale)

            # update ts_slot difference in case of combined time slots
            t_now_diff = 1
            if self.combine is not None:
                if t_now_slot == min(self.combine):
                    t_now_diff = max(self.combine) - min(self.combine) + 1

            # distribute n_arrivals uniformly
            if n_arrivals > 0.5:
                arrivals = np.linspace(0, t_now_diff, n_arrivals + 2)[1:-1]
                # add n arrivals to ts and update t_now
                ts.extend(np.array(arrivals + t_now_slot - 1).astype(float))
            t_now = t_now + t_now_diff

        print('Generated !')

        return np.array(ts) , t_now, self._scale_min
