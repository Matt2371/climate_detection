import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
import os.path


# tqdm wraps iterables and shows progress bar

### Single and multiple scenario analysis (see README.txt) of SSJRB model outputs of reservoir performance objectives
### and detects significance of projected outputs compared to historical baseline

## Shared parameters:
## objective: ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs'], str
## win_size: moving average window size applied, int
## alt = ['two-sided', 'less', 'greater']

def remove_lag1(X):
  """
  Remove lag-1 autocorrelation using pre-whitening method
  X - numpy array - input sequence
  Returns: pre-whitened sequence
  """
  
  # Estimate lag1 autocorrelation
  rho1 = acf(X)[1]
  
  # Apply pre-whitening using estimated lag-1 autocorrelation
  Y = np.zeros(len(X))
  for i in range(1, len(X)):
    Y[i] = X[i] - rho1*X[i-1]

  return Y


def rolling_significance(gcm, rcp, lulc, objective, parametric=False, alt='two-sided', win_size=30, pre_whitening=False):
    """
    take rolling averages, run statistical significance tests against historical
    Parameters: 
    gcm/rcp (from cmip5),
    lulc(land use), 
    objective (['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs']),
    parametric (t-test or MWU), 
    alt ('greater/less/two-sided' hypothesis compared to historical), 
    win_size(size of MA window)
    pre_whitening (boolean, whether or not to apply pre-whitening method to remove lag-1 autocorrelation)
    
    returns: dataframe with original data AND p-values
    """

    # load dataframe and isolate objective
    # "scenario" stores cmip5 name
    scenario = gcm + '_' + rcp + '_r1i1p1'
    df = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0, parse_dates=True)
    df = df[[objective]]

    # remove lag-1 autocorrelation if pre_whitening is True
    if pre_whitening:
        values = remove_lag1(df.values.flatten())
        df[objective] = values
    
    # set historical df's
    his_df = df['1951-10-01':'2000-10-01'].copy()
    # set projection df based on window size
    lower_yr = str(2000 - win_size + 1)
    proj_df = df[lower_yr + '-10-01':'2098-10-01'].copy()

    # get rolling samples (future), delete anything before year 2000
    proj_df['rolling'] = [window.to_list() for window in proj_df.loc[:, objective].rolling(win_size)]
    proj_df.loc[lower_yr + '-10-01':'1999-10-01', 'rolling'] = float("NaN")

    # iterate over rolling samples and conduct significance test. "row" stores (index (date), objective data,
    # rolling samples) as named tuple
    for row in proj_df.itertuples():
        # conduct t-tests
        if parametric:
            df.loc[row[0], objective + '_p-value'] = st.ttest_ind(row[2], his_df, alternative=alt, equal_var=False)[1]
        else:
            # conduct MWU, skip if any NaN cells in window
            if np.isnan(row[2]).any():
                continue
            else:
                df.loc[row[0], objective + '_p-value'] = st.mannwhitneyu(row[2], his_df[objective], alternative=alt,
                                                                         use_continuity=True)[1]
    return df


# store names of gcm/rcp/lulc combinations
gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
            'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
            'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
            'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
            'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']

rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
lulc_names = pd.read_csv('lulc_scenario_names.csv').name.to_list()



def export_agg(objective, parametric=False, alt='two-sided', win_size=30):
    '''
    returns tables with p values aggregated for every scenario (gcm/rcp/lulc combination)
    parameters: same as definitions as SHARED PARAMETERS
    '''

    # import aggregate csv's (empty, dates only)
    agg_all = pd.read_csv('empty/datetime.csv', index_col=0,
                          parse_dates=True)

    # conduct rolling significance for all gcm/rcp combinations. aggregate results(p-vals) into df's
    for gcm in tqdm(gcm_list, desc='Getting p-vals'):
        for rcp in rcp_list:
            for lulc in lulc_names:
                try:
                    p_vals = rolling_significance(gcm, rcp, lulc, objective, parametric, alt, win_size)[
                        objective + '_p-value']
                    agg_all[gcm + '_' + rcp + '_' + lulc] = p_vals

                except FileNotFoundError:
                    pass

    return agg_all



def p_val_count(df):
    '''
    for aggregate df's (see function above), add "count" column that keeps tract of p<0.05 for each datetime (multiple
    scenario)
    returns: inputted dataframe with added "count" column
    '''

    # iterate over df rows. "row" stores (index (date), data from each columns) as named tuple
    for row in df.itertuples():
        # p < 0.05 counter, reset counter for every new row
        n = 0
        for i in range(len(row) - 1):
            # add to counter if p<= 0.05, i+1 to skip datetime
            if row[i + 1] <= 0.05:
                n = n + 1
            else:
                continue
        df.loc[row[0], 'count'] = n

    return df



def first_significance(df):
    '''
    returns: year of first significance for each model scenario (single scenario)
    parameter: input df with p values
    '''

    # create list of columns (names of each scenario)
    columns = list(df)

    # initialize list of years of first significance
    date = []

    # check one column at a time
    for col in columns:
        # check if significance can actually be detected
        if any(x < 0.05 for x in df[col]):
            pass
        else:
            date.append(float('NaN'))

        # iterate over rows for single col
        # 'row' stores [datetime, col p value]
        for row in df[[col]].itertuples():
            if row[1] < 0.05:
                # store first year, stop loop (so no later values are stored)
                date.append(row[0])
                break
            else:
                continue
    # create output df
    data = {'Model': columns, 'Date': date}
    output = pd.DataFrame(data)

    # reformat dates as year
    # if not dates found, ignore AttributeError
    try:
        output['Year'] = output['Date'].dt.year
        output = output.drop(['Date'], axis=1)
    except AttributeError:
        output['Year'] = float('NaN')
        pass

    # print(output)
    return output



def expanding_significance(gcm, rcp, lulc, objective='Upstream_Flood_Volume_taf', alt='greater',
                           min_periods=30):
    '''
    conduct MWU test on expanding window of FLODO VOLUME (will this lead to more montonic detection?)
    min periods is the minimum number of units before calculations of expanding window begins
    '''

    # load scenario
    scenario = gcm + '_' + rcp + '_r1i1p1'
    df = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0, parse_dates=True)
    df = df[[objective]]

    # set historical df's
    his_df = df['1951-10-01':'2000-10-01'].copy()
    # set projection df based on minimum period size
    lower_yr = str(2000 - min_periods + 1)
    proj_df = df[lower_yr + '-10-01':'2098-10-01'].copy()

    # get expanding samples (future), delete anything before year 2000
    proj_df['expanding'] = [window.to_list() for window in proj_df.loc[:, objective].expanding(min_periods)]
    proj_df.loc[lower_yr + '-10-01':'1999-10-01', 'expanding'] = float("NaN")

    # iterate over expanding windows and conduct MWU test
    for row in proj_df.itertuples():
        # conduct MWU, skip if any NaN cells in window
        if np.isnan(row[2]).any():
            continue
        else:
            df.loc[row[0], objective + '_p-value'] = st.mannwhitneyu(row[2], his_df[objective], alternative=alt,
                                                                     use_continuity=True)[1]
    return df



def expanding_export_agg(objective='Upstream_Flood_Volume_taf'):
    '''
    get p-vals for expanding flooding MWU tests
    parameters: same as definitions as SHARED PARAMETERS
    '''

    # import aggregate csv's (empty, dates only)
    index = pd.read_csv('empty/datetime.csv', index_col=0,
                              parse_dates=True).index
    # initiate dictionary to store p-vals for each scenario
    data = {}
    # conduct expanding significance for all gcm/rcp combinations. aggregate results(p-vals) into df's
    for gcm in tqdm(gcm_list, desc='Getting p-vals'):
        for rcp in rcp_list:
            for lulc in lulc_names:
                try:
                    p_vals = expanding_significance(gcm=gcm, rcp=rcp, lulc=lulc)[objective + '_p-value'].values
                    data[gcm + '_' + rcp + '_' + lulc] = p_vals

                except FileNotFoundError:
                    pass
    # build export dataframe
    exp_agg_all = pd.DataFrame(data, index=index)
    return exp_agg_all

