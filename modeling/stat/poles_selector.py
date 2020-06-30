# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# Class for pole seletor. needed to select the number of poles to be used in analysis
#
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import pandas as pd

class poles_selector:
    def __init__(self,alldata,year):
        self._year = year
        self._alldata = alldata
        self._data = self._alldata[self._alldata['Start_year'].isin(year)]

    def select_poles(self,by,topn=100,eachn=30,currenty=2015):
        self._topn = topn
        self._eachn = eachn
        self._current_year = currenty
        if by == 'topn':
            self.select_topn()
        if by == 'onceeachn':
            self.select_eachn()
        if by == 'continous':
            self.select_continous()


    def select_topn(self):
        # top n poles
        n = self._topn
        pole_freq = self._data.groupby(['Start_year', 'Charge_point']).agg(['count']).reset_index().iloc[:, 0:3]
        pole_freq.columns = ['Start_year', 'Pole', 'Counts']
        pole_freq = pole_freq.sort_values(by='Counts', ascending=False).reset_index()
        charge_points = pole_freq['Pole'][0:n]
        self._charge_points = charge_points


    def select_eachn(self):
        # poles used each n days once
        n_day = self._eachn
        every_what = 'Start_DOY'
        n_fac = np.floor(365 / n_day) + 1
        s = n_fac * (n_fac + 1) / 2
        ts_d_temp = self._data.copy()
        ts_d_temp[every_what] = np.floor(ts_d_temp[every_what] / n_day) + 1
        pole_freq = ts_d_temp.groupby(['Start_year', every_what, 'Charge_point']).agg(['count']).reset_index().iloc[:,
                    0:4]
        pole_freq.columns = ['Start_year', every_what, 'Pole', 'Counts']
        pole_freq_1 = pole_freq.groupby(['Start_year', 'Pole']).agg(['sum']).reset_index().iloc[:, 0:3]
        pole_freq_1.columns = ['Start_year', 'Pole', 'sum']
        pole_freq_1 = pole_freq_1.loc[pole_freq_1['sum'] == s]
        charge_points = pole_freq_1['Pole']  # poles used each 1 day=0 each 2 days=4 each 7 days=278 each 15 days=708
        self._charge_points = charge_points

    def select_continous(self):
       previous_data = self._alldata[self._alldata['Start_year'].isin(  [self._year[0]-1]  )]
       previous_poles = np.unique(previous_data['Charge_point'])

       next_data = self._alldata[self._alldata['Start_year'].isin(  [self._year[0] + 1]  )]
       next_poles = np.unique(next_data['Charge_point'])

       charge_points = list(set(previous_poles).intersection(next_poles))
       self._charge_points = pd.Series( np.array( charge_points) )




















