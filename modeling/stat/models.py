# -------------------------------------------------- ML 02/10/2019 ----------------------------------------------------#
#
# here we have models. poly fit and loess and other models that are being used to estimate lambda
# -------------------------------------------------------------------------------------------------------------------- #



from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import math
from scipy.optimize import newton
from scipy.special import digamma


import pandas as pd
from sklearn.metrics.scorer import make_scorer

def cude_error(y_true, y_pred):
    # sum_cube_abs = np.mean(np.exp(np.abs(y_true - y_pred)))
    sum_cube_abs = np.abs((y_true - y_pred)**2).mean()
    return sum_cube_abs
my_scorer = make_scorer(cude_error, greater_is_better=False)

def best_poly_fit(y,x,max_deg,verbose=1, scoring = 'neg_root_mean_squared_error'):
    # reshaping the input and output variables to have appropriate shape
    y = y.reshape(-1, 1)
    try:
        print('Number of features:', x.shape[1])
    except:
        x = x.reshape(-1, 1)

    if scoring == 'my_scorer':
        scoring = my_scorer
    # this preprocesses the data
    # numeric_features = ['age', 'fare']
    # numeric_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('scaler', StandardScaler())])
    # categorical_features = ['embarked', 'sex', 'pclass']
    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer)
    #         #('cat', categorical_transformer)
    #     ])

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))



    # clf = Pipeline(steps=[#('preprocessor', preprocessor),
    #                       ('linearregression', PolynomialRegression())])
    #
    # param_grid = {#'preprocessor__num__imputer__strategy': ['mean'],
    #               'linearregression__polynomialfeatures__degree': np.arange(10),
    #               'linearregression__fit_intercept': [True, False],
    #               'linearregression__normalize': [True, False]}

    param_grid = {'polynomialfeatures__degree': np.arange(max_deg)}
    if verbose==0: verbose = 1
    poly_grid = GridSearchCV(PolynomialRegression(), param_grid,
                             cv=10,
                             scoring=scoring,
                             verbose=verbose-1)
    # doing grid search
    poly_grid.fit(x,y)
    # fit on the best parameters
    poly_grid.best_estimator_.fit(x,y)

    pred = poly_grid.predict(x)

    var = np.var(pred-y)
    poly_grid._total_var = var
    poly_grid._total_std = np.sqrt(var)

    return poly_grid

def get_gpr_fit(y,x):
    # reshaping the input and output variables to have appropriate shape
    y = y.reshape(-1, 1)
    try:
        print('Number of features:', x.shape[1])
    except:
        x = x.reshape(-1, 1)

    gpr_fit = GaussianProcessRegressor(random_state = 0).fit(x, y)
    # print(gpr_fit.score(x, y))

    # aaaa = gpr_fit.predict(x, return_std=True)

    return gpr_fit

def loc_eval(x, b):
    """
    Evaluate `x` using locally-weighted regression parameters.
    Degree of polynomial used in loess is inferred from b. `x`
    is assumed to be a scalar.
    """
    loc_est = 0
    for i in enumerate(b): loc_est+=i[1]*(x**i[0])
    return(loc_est)

def loess_fit(xvals, yvals, alpha, poly_degree=1, robustify=False):

    """
    link - http://www.jtrive.com/loess-nonparametric-scatterplot-smoothing-in-python.html#Footnotes:
    Perform locally-weighted regression via xvals & yvals.
    Variables used within `loess` function:

        n         => number of data points in xvals
        m         => nbr of LOESS evaluation points
        q         => number of data points used for each
                     locally-weighted regression
        v         => x-value locations for evaluating LOESS
        locsDF    => contains local regression details for each
                     location v
        evalDF    => contains actual LOESS output for each v
        X         => n-by-(poly_degree+1) design matrix
        W         => n-by-n diagonal weight matrix for each
                     local regression
        y         => yvals
        b         => local regression coefficient estimates.
                     b = `(X^T*W*X)^-1*X^T*W*y`. Note that `@`
                     replaces np.dot in recent numpy versions.
        local_est => response for local regression
    """
    # sort dataset by xvals:
    all_data = sorted(zip(xvals, yvals), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)

    locsDF = pd.DataFrame(
                columns=[
                  'loc','x','weights','v','y','raw_dists',
                  'scale_factor','scaled_dists'
                  ])
    evalDF = pd.DataFrame(
                columns=[
                  'loc','est','b','v','g'
                  ])

    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = max(0,min(xvals)-(.5*avg_interval))
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)

    # Generate design matrix based on poly_degree.
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T


    for i in v:

        iterpos = i[0]
        iterval = i[1]

        # Determine q-nearest xvals to iterval.
        iterdists = sorted([(j, np.abs(j-iterval)) \
                           for j in xvals], key=lambda x: x[1])

        _, raw_dists = zip(*iterdists)

        # Scale local observations by qth-nearest raw_dist.
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 \
                      if j[1]<=1 else 0)) for j in scaled_dists]

        # Remove xvals from each tuple:
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))

        iterDF1 = pd.DataFrame({
                    'loc'         :iterpos,
                    'x'           :xvals,
                    'v'           :iterval,
                    'weights'     :weights,
                    'y'           :yvals,
                    'raw_dists'   :raw_dists,
                    'scale_fact'  :scale_fact,
                    'scaled_dists':scaled_dists
                    })

        locsDF = pd.concat([locsDF, iterDF1])
        W = np.diag(weights)
        y = yvals
        b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        local_est = loc_eval(iterval, b)
        try:
            actual_y = y[iterpos]
            diff = local_est-actual_y
        except:
            diff = 0

        iterDF2 = pd.DataFrame({
                     'loc':[iterpos],
                     'b'  :[b],
                     'v'  :[iterval],
                     'g'  :[local_est],
                     'diff':[diff]
                     })

        evalDF = pd.concat([evalDF, iterDF2])

    # Reset indicies for returned DataFrames.
    locsDF.reset_index(inplace=True)
    locsDF.drop('index', axis=1, inplace=True)
    locsDF['est'] = 0; evalDF['est'] = 0
    locsDF = locsDF[['loc','est','v','x','y','raw_dists',
                     'scale_fact','scaled_dists','weights']]


    if robustify==True:

        cycle_nbr = 1
        robust_est = [evalDF]

        while True:
            # Perform iterative robustness procedure for each local regression.
            # Evaluate local regression for each item in xvals.
            #
            # e1_i => raw residuals
            # e2_i => scaled residuals
            # r_i  => robustness weight
            revalDF = pd.DataFrame(
                            columns=['loc','est','v','b','g']
                            )

            for i in robust_est[-1]['loc']:

                prevDF = robust_est[-1]
                locDF = locsDF[locsDF['loc']==i]
                b_i = prevDF.loc[prevDF['loc']==i,'b'].item()
                w_i = locDF['weights']
                v_i = prevDF.loc[prevDF['loc']==i, 'v'].item()
                g_i = prevDF.loc[prevDF['loc']==i, 'g'].item()
                e1_i = [k-loc_eval(j,b_i) for (j,k) in zip(xvals,yvals)]
                e2_i = [j/(6*np.median(np.abs(e1_i))) for j in e1_i]
                r_i = [(1-np.abs(j**2))**2 if np.abs(j)<1 else 0 for j in e2_i]
                w_f = [j*k for (j,k) in zip(w_i, r_i)]    # new weights
                W_r = np.diag(w_f)
                b_r = np.linalg.inv(X.T @ W_r @ X) @ (X.T @ W_r @ y)
                riter_est = loc_eval(v_i, b_r)

                riterDF = pd.DataFrame({
                             'loc':[i],
                             'b'  :[b_r],
                             'v'  :[v_i],
                             'g'  :[riter_est],
                             'est':[cycle_nbr]
                             })

                revalDF = pd.concat([revalDF, riterDF])
            robust_est.append(revalDF)

            # Compare `g` vals from two latest revalDF's in robust_est.
            idiffs = \
                np.abs((robust_est[-2]["g"]-robust_est[-1]["g"])/robust_est[-2]["g"])

            if ((np.all(idiffs<.005)) or cycle_nbr>50): break

            cycle_nbr+=1

        # Vertically bind all DataFrames from robust_est.
        evalDF = pd.concat(robust_est)

    evalDF.reset_index(inplace=True)
    evalDF.drop('index', axis=1, inplace=True)
    evalDF = evalDF[['loc','est', 'v', 'b', 'g','diff']]
    evalDF['b_str'] = evalDF['b'].astype(str)
    std_diff = evalDF.groupby('b_str').std().fillna(0).reset_index() # evalDF.groupby('b_str').last()
    joinleft = pd.DataFrame({'b_str':std_diff['b_str'],'std':std_diff['diff']})
    evalDF = pd.merge(evalDF,joinleft,on='b_str', how='left')
    return(locsDF, evalDF)

def fit_neg_binom(vec, init=0.0001):
    def r_derv(r_var, vec):
        ''' Function that represents the derivative of the neg bin likelihood wrt r
        @param r: The value of r in the derivative of the likelihood wrt r
        @param vec: The data vector used in the likelihood
        '''
        if not r_var or not vec.any():
            raise ValueError("r parameter and data must be specified")

        if r_var <= 0:
            raise ValueError("r must be strictly greater than 0")

        total_sum = 0
        obs_mean = np.mean(vec)  # Save the mean of the data
        n_pop = float(len(vec))  # Save the length of the vector, n_pop

        for obs in vec:
            total_sum += digamma(obs + r_var)

        total_sum -= n_pop*digamma(r_var)
        total_sum += n_pop*math.log(r_var / (r_var + obs_mean))

        return total_sum

    def p_equa(r_var, vec):
        ''' Function that represents the equation for p in the neg bin likelihood wrt p
        @param r: The value of r in the derivative of the likelihood wrt p
        @param vec: Te data vector used in the likelihood
        '''
        if not r_var or not vec.any():
            raise ValueError("r parameter and data must be specified")

        if r_var <= 0:
            raise ValueError("r must be strictly greater than 0")

        data_sum = np.sum(vec)
        n_pop = float(len(vec))
        p_var = 1 - (data_sum / (n_pop * r_var + data_sum))
        return p_var

    def neg_bin_fit(vec, init=0.00001):
        ''' Function to fit negative binomial to data
        @param vec: The data vector used to fit the negative binomial distribution
        @param init: Set init to a number close to 0, and you will always converge
        '''
        if not vec.any():
            raise ValueError("Data must be specified")

        est_r = newton(r_derv, init, args=(vec,))
        est_p = p_equa(est_r, vec)
        return est_r, est_p

    try:
        return neg_bin_fit(vec=vec,init=init)
    except:
        return 0,0