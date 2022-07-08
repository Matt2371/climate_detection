import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# store gcm/rcp names
gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
            'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
            'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
            'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
            'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']

rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']


# create aggregate data dfs of all scenarios, also organized by rcp
# parameters: objective('flood'/'reliability')
# returns: df's [all, rcp26, rcp45, rcp60, rcp85]
def agg_multiple(objective):
    # build df's w/ aggregated (climate data) and sorted by RCP, start from empty
    agg_all = pd.read_csv('empty/datetime.csv', index_col=0,
                          parse_dates=True)
    agg_rcp26 = pd.read_csv('empty/datetime.csv', index_col=0,
                            parse_dates=True)
    agg_rcp45 = pd.read_csv('empty/datetime.csv', index_col=0,
                            parse_dates=True)
    agg_rcp60 = pd.read_csv('empty/datetime.csv', index_col=0,
                            parse_dates=True)
    agg_rcp85 = pd.read_csv('empty/datetime.csv', index_col=0,
                            parse_dates=True)
    # fill in data to tables
    for gcm in gcm_list:
        for rcp in rcp_list:
            try:
                scenario = gcm + '_' + rcp + '_r1i1p1'
                df = pd.read_csv('data/%s-results.csv' % scenario, index_col=0, parse_dates=True)

                agg_all[gcm + '_' + rcp] = df[objective]

                if rcp == 'rcp26':
                    agg_rcp26[gcm + '_' + rcp] = df[objective]
                elif rcp == 'rcp45':
                    agg_rcp45[gcm + '_' + rcp] = df[objective]
                elif rcp == 'rcp60':
                    agg_rcp60[gcm + '_' + rcp] = df[objective]
                elif rcp == 'rcp85':
                    agg_rcp85[gcm + '_' + rcp] = df[objective]

            except FileNotFoundError:
                pass
    return agg_all, agg_rcp26, agg_rcp45, agg_rcp60, agg_rcp85


# add p-value to inputted df based on significance test against set historical year
# parameters: df, historical_datetime, parametric (True/t-test, False/MWU)
def multiple_significance(df, historical_datetime='1971-10-01', parametric=True):
    # iterate over "future" datetime (itertuple[0]), conduct significance tests
    future_df = df.loc['2000-10-01':'2098-10-01', :].copy()

    for row in future_df.itertuples():
        if parametric:
            future_df.loc[row[0], 'p-value'] = st.ttest_ind(df.loc[historical_datetime, :], df.loc[row[0], :],
                                                            equal_var=False)[1]

        else:
            future_df.loc[row[0], 'p-value'] = st.mannwhitneyu(df.loc[historical_datetime, :], df.loc[row[0], :],
                                                               use_continuity=True, alternative='two-sided')[1]

    return future_df


df = agg_multiple('reliability')[0]
print(multiple_significance(df))
