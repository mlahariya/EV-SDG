# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# here we defined the beta mixture model
#
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta,norm
from handles.data_hand import get_slotted_data
import matplotlib.pyplot as plt
import random
from sklearn import mixture
random.seed( 30 )


# mixture optimizer for same class mixtures
class mixture_optimizer():
    # for beta distribution: a = alpha ; b = beta
    #   here both are bounded to be > 0
    # for normal distribution a = mean; b = std
    #   here we only impose the condition b(std) > 0

    def __init__(self,data,type):
        self._data = self.check_data(data)
        self.prepare_epdf()
        self.model_type = type

    def check_data(self,data):
        data = data[np.logical_and(data >= 0.0, data <= 1.0)]
        return np.array(data).flatten()

    def prepare_epdf(self):
        self._bin_size = 100
        bins = np.linspace(0, 1, self._bin_size)
        digitized = np.digitize(self._data, bins)
        bin_counts = np.array([len(self._data[digitized == i]) for i in range(1, len(bins))]).flatten()
        self._x_values = bins[:-1]  +(1/ (2*self._bin_size))
        self._epdf = bin_counts/len(self._data)

    def define_unknowns(self,n_modes, init_b = np.array([ 50 ]), init_a = np.array([ 0.5])):
        self._n_modes = n_modes
        weights = np.array(np.ones(n_modes)/n_modes).flatten()
        bs = np.array(np.ones(n_modes) * init_b).flatten()
        a_s = np.array(np.ones(n_modes) * init_a).flatten()
        self._initial_unknowns = np.concatenate((weights,bs,a_s))
        self._final_unknowns = self._initial_unknowns.copy()

    def pdf_mm(self,constraints = None,x = None):
        training = True
        if constraints is None:
            constraints = self._final_unknowns.copy()
        if x is None:
            x = self._x_values.copy()
            training = True
        else:
            training = True
        su = 0
        for i in np.arange(self._n_modes):
            weight = constraints[i]
            b = constraints[i+self._n_modes]
            a = constraints[i+ (self._n_modes*2)]

            if self.model_type == 'beta':
                prob = beta(a, b).pdf(x)

            if self.model_type == 'normal':
                prob = norm(a,b).pdf(x)

            if training:
                prob = prob/len(x)


            su = su + prob*weight
        # su = su/sum(su)
        return su

    def distance(self,x):
        sub = self.pdf_mm(constraints=x) - self._epdf
        z = np.sqrt( np.square(sub).sum()/len(sub) )
        return z

    def weights(self,x):
        return x[0:self._n_modes].sum() - 1

    def ineq_weights(self,x):
        ws = x[0:self._n_modes]
        return int(all(i > 0 for i in ws)) - 0.5

    def ineq_b(self,x):
        bs = x[self._n_modes:self._n_modes*2]
        return int(all(i > 0 for i in bs)) - 0.5

    def ineq_a(self,x):
        a_s = x[self._n_modes*2:self._n_modes*3]
        return int(all(i > 0 for i in a_s)) - 0.5

    def optimize(self,verbose=0,printer=""):

        if self.model_type == 'beta':

            cons = [{'type': 'eq', 'fun': self.weights},
                    # {'type': 'ineq', 'fun': self.ineq_weights},
                    {'type': 'ineq', 'fun': self.ineq_b},
                    {'type': 'ineq', 'fun': self.ineq_a}]

        if self.model_type == 'normal':

            cons = [{'type': 'eq', 'fun': self.weights},
                    {'type': 'ineq', 'fun': self.ineq_weights},
                    {'type': 'ineq', 'fun': self.ineq_b}]

        display = bool(verbose-1)

        res = minimize(self.distance, self._final_unknowns, constraints=cons,
                       options= {'maxiter':1000,'disp':display})
        if not(res.success) and bool(verbose):
            print(('FAILED','red'),': Optimization failed for ' +str(printer))
        self.optimization_result = res
        self._final_unknowns = res['x'].copy()

    def plot(self):
        x= self._x_values
        y = self.pdf_bmm(constraints=self._final_unknowns)
        print(sum(y))
        plt.plot(x, self._epdf)
        plt.plot(x, y)
        return plt

    def rand_samp(self,size=1):
        n_modes = self._n_modes
        mixing = self._final_unknowns[0:n_modes]
        std = self._final_unknowns[n_modes:(2 * n_modes)]
        means = self._final_unknowns[(2 * n_modes):(3 * n_modes)]
        samples = []
        if not (sum(np.isnan(mixing)) > 0):
            mixing[mixing<0] = 0
            mixing = mixing/sum(mixing)
            for i in range(0, size):
                comp = np.random.choice(range(0, len(mixing)), p=mixing, size=1)
                m = means[comp]
                st = std[comp]
                samp = -1
                while samp <= 0 or samp > 1:
                    if self.model_type == 'normal':
                        samp = np.random.normal(loc=m, scale=st, size=1)
                    if self.model_type == 'beta':
                        samp = np.random.beta(a=m, b=st, size=1)
                samples.append(samp)
        else:
            samples = np.ones(size)*0.5
        return np.array(samples)

class mixture_models:

    def __init__(self,y,x,initilizations,sesonality=24.00,x_meta=None,combine=None):
        self.model_for = y.name
        self.x_names = np.array( x.columns )
        self.y = np.array(y)
        self.x = np.array(x)
        self.initilizations = initilizations
        self.x_meta=x_meta
        self.sesonality = float( sesonality )
        self.processed_data = self.get_combined_y_data(combine=combine)

    def combine_timeslots(self,x,combine):
        p = x.copy()
        p[np.in1d(x, combine)] = combine[0]
        return p

    def reset_scale_multiplier(self):
        self._scale_multiplier = 1

    def get_processed_data(self):
        y = self.y.reshape(-1, 1)
        fin_d = pd.DataFrame(np.concatenate((y, self.x), axis=1))
        fin_d.columns = np.concatenate(
            (np.array('y').reshape(-1, 1), np.array(self.x_names).reshape(-1, 1)), axis=0).flatten()
        fin_d['y'] = pd.to_numeric(fin_d['y'])
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

    def get_combined_y_data(self,combine):
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
        self.processed_data['initilizations'] = np.array(self.initilizations)

        return self.processed_data

    # ------------------------------------------- FITTING --------------------------------------------------------------

    def get_intital_ab(self,mu, var):
        alpha = ((1 - mu) / var - (1 / mu)) * (mu ** 2)
        beta = alpha * ((1 / mu) - 1)
        return alpha,beta

    def fit_mm_mindist(self,data,log=False,verbose=0):
        # in final unknowns, the values are seperated like
        # mixing components, b_values, a_values
        # initilization
        n_clust = len(np.unique(data['initilizations']))
        means = np.array(data.groupby(['initilizations'])['y'].mean()).flatten()
        std = np.array(data.groupby(['initilizations'])['y'].std()).flatten()
        weights = np.array(data.groupby(['initilizations'])['y'].count()).flatten()/len(data['initilizations'])
        means[np.isnan(means)] = 0
        std[np.isnan(std)] = 0.001
        precisions = std ** (-2)
        X = np.array(data['y'])
        if log:
            X_fit = np.log(X)
        else:
            X_fit = X
        X_fit = X_fit[~np.isnan(X_fit)].reshape(-1, 1)
        f1 = np.unique(data[self.x_names[0]])
        f2 = np.unique(data[self.x_names[1]])


        # fitting
        if len(X_fit) > 0:
            # Fit a Gaussian mixture with EM using ten components
            if self._mix == 'beta':

                ab = np.array([self.get_intital_ab(mu=means[i], var=precisions[i] ** -1) for i in np.arange(n_clust)])
                mm = mixture_optimizer(data=data['y'], type = self._mix)
                mm.define_unknowns(n_modes=n_clust, init_a= np.array( ab[:,0] ).flatten(), init_b= np.array( ab[:,1] ).flatten() )
                mm.optimize(verbose=verbose)
                return mm, f1, f2

            if self._mix == 'normal':
                mm = mixture_optimizer(data=data['y'], type = self._mix)
                mm.define_unknowns(n_modes=n_clust, init_a=np.array(means).flatten(),
                                   init_b=np.array(std).flatten())
                mm.optimize(verbose=verbose,printer=str(f1) + " " + str(f2))
                return mm, f1, f2
        return None, f1, f2

    def fit_mm_EM(self,data,log=False,verbose=0):
        # initilization
        if self._n_mix is None:
            n_clust = len(np.unique(data['initilizations']))
            means = np.array(data.groupby(['initilizations'])['y'].mean()).reshape(-1, 1)
            std = np.array(data.groupby(['initilizations'])['y'].std()).flatten()
        else:
            n_clust = self._n_mix
            means = (np.ones(self._n_mix)*0.5).reshape(-1, 1)
            std = (np.ones(self._n_mix)*0.05).flatten()
        precisions = std ** (-2)
        X = np.array(data['y'])
        if log:
            X_fit = np.log(X)
        else:
            X_fit = X
        X_fit = X_fit[~np.isnan(X_fit)].reshape(-1, 1)
        means[np.isnan(means)] = 0
        precisions[np.isnan(precisions)] = 0.001

        f1 = np.unique(data[self.x_names[0]])
        f2 = np.unique(data[self.x_names[1]])

        # fitting
        if len(X_fit) > 1:
            # Fit a Gaussian mixture with EM using ten components
            gmm = mixture.GaussianMixture(n_components=n_clust, covariance_type='spherical',
                                          max_iter=1000, means_init=means, precisions_init=precisions).fit(X_fit)
            return gmm, f1,f2

        return None, f1, f2

    def fit(self,mix='normal',method = 'mindist',combine=None,initilizations = None,verbose=0,n_mix=None):

        # ------------------------------ Prepration -------------------------------------------------
        # create dataset for modeling
        # if continous = True, we replace the values of means with continous values
        self._mix = mix
        self._n_mix = n_mix
        self._method = method
        if initilizations is not None:
            self.initilizations = initilizations
        if combine is None:
            combine = self.combine


        # pull the processed dataset after passingon the combine argument
        fin_d = self.get_combined_y_data(combine=combine)

        # ------------------------------ Modeling -------------------------------------------------
        model_data = fin_d['y'].copy()
        fac1 = fin_d[self.x_names[0]]
        orignal_start_slot = fin_d[self.x_names[1]]
        fit = []
        # put bounds and normalize
        fin_d = fin_d[model_data < 24]
        fin_d['y'] = fin_d['y']/24

        for f1 in np.unique(fac1):
            d = fin_d[(fac1 == f1)]
            fac2 = fin_d['Start_time_internal']
            for f2 in np.unique(fac2):
                d2 = d.loc[d.Start_time_internal == f2]
                if verbose > 1: print(' \t\t Fitting parameters for factor : ', str(f1), ' timeslot: ',f2)
                if method == 'mindist':
                    temp_fit,f1_d,f2_d = self.fit_mm_mindist(data=d2, verbose=verbose)
                if method == 'EM':
                    temp_fit, f1_d, f2_d = self.fit_mm_EM(data=d2, verbose=verbose)

                # saving models for each time slot
                for f2_d_loop in f2_d:
                    fit.append([f1_d[0],int(f2_d_loop), temp_fit])
                    # names to be added to final fitted variable
                    names = np.append(self.x_names, 'Model')

        fit_merged = [list(x) for x in zip(*fit)]
        fit_named = dict(zip(names, fit_merged))
        self._best_fit = fit_named
        return fit_named

    # ------------------------------------------- SAMPLE --------------------------------------------------------------

    def pop_model(self,f1,f2):
        # this function returns the poly model for the given f1
        # this method can be called for both poly and loess models
        # returns a tuple, that means we have to select the first element to get the model
        # tuple has model, y, and x values
        selector_boolan = ((np.array(self._best_fit[self.x_names[0]]) == f1) & (np.array(self._best_fit[self.x_names[1]]) == f2))
        selector = np.where(selector_boolan.flatten())[0]
        Model = self._best_fit['Model']
        M_topop = Model[int(selector)]
        return M_topop

    def predict_mix(self,f1,f2):
        model = self.pop_model(f1=f1, f2=f2)
        if self._method =='mindist':
            sample = model.rand_samp(size=1)
        if self._method =='EM':
            sample = -1
            while sample <=0 or sample >=1:
                sample = float(model.sample(1)[0])


        return sample*self.scale

    def predict_day(self, X_test, arrivals, slot,verbose=0):

        # here we generate a days time series
        if verbose > 2: print(' \t\t Sampling from MM for : ', str(np.array( X_test)[0][0]))
        arrivals = arrivals
        deps = []
        self.scale = 24

        for arr in arrivals:

            t_now_slot = int(get_slotted_data(data=arr,slot_secs= slot))

            dep = self.predict_mix(f1=np.array(X_test),f2=t_now_slot)

            deps.append(float(dep))

        return np.array(deps)