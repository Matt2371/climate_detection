import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


# take rolling averages, run statistical significance tests against historical
# parameters: gcm, rcp, objective (reliability/flood), parametric (t-test or MWU)
# returns: dataframe with original data AND p-values
def rolling_significance(gcm, rcp, objective, parametric=True):
    # load dataframe and isolate objective
    scenario = gcm + '_' + rcp + '_r1i1p1'
    df = pd.read_csv('data/%s-results.csv' % scenario, index_col=0, parse_dates=True)
    df = df[[objective]]

    # set historical and projection df's
    his_df = df['1951-10-01':'2000-10-01'].copy()
    proj_df = df['1971-10-01':'2098-10-01'].copy()

    # get rolling samples (future), delete anything before year 2000
    proj_df['rolling'] = [window.to_list() for window in proj_df.loc[:, objective].rolling(30)]
    proj_df.loc['1971-10-01':'1999-10-01', 'rolling'] = float("NaN")

    # iterate over rolling samples and conduct significance test. "row" stores (index (date), objective data,
    # rolling samples) as named tuple
    for row in proj_df.itertuples():
        # conduct t-tests
        if parametric:
            df.loc[row[0], objective + '_p-value'] = st.ttest_ind(his_df, row[2], equal_var=False)[1]
        else:
            # conduct MWU, skip if any NaN cells in window
            if np.isnan(row[2]).any():
                continue
            else:
                df.loc[row[0], objective + '_p-value'] = st.mannwhitneyu(his_df[objective], row[2], use_continuity=True,
                                                                         alternative='two-sided')[1]
    return df


# for aggregate df's (see fun below below), add "count" column that keeps tract of p<0.05 for each datetime
# returns: inputted dataframe with added "count" column
def p_val_count(df):
    # create list of columns
    columns = list(df)

    # iterate over df rows. "row" stores (index (date), columns of samples) as named tuple
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


# store names of gcm/rcp combinations
gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
            'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
            'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
            'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
            'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']

rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']


# export and plot results for every gcm/rcp combination (single scenario)
# parameters: objective ('reliability'/'flood'), parametric=True runs t-test, False runs MWU
def export_plt(objective, parametric=True):
    # import aggregate csv's (empty, dates only)
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
    # conduct rolling significance for all gcm/rcp combinations. aggregate results(p-vals) into df's
    for gcm in gcm_list:
        for rcp in rcp_list:
            try:
                p_vals = rolling_significance(gcm, rcp, objective, parametric)[objective + '_p-value']
                agg_all[gcm + '_' + rcp] = p_vals

                # add p-vals to rcp specific df's as well
                if rcp == 'rcp26':
                    agg_rcp26[gcm + '_' + rcp] = p_vals
                elif rcp == 'rcp45':
                    agg_rcp45[gcm + '_' + rcp] = p_vals
                elif rcp == 'rcp60':
                    agg_rcp60[gcm + '_' + rcp] = p_vals
                elif rcp == 'rcp85':
                    agg_rcp85[gcm + '_' + rcp] = p_vals

            except FileNotFoundError:
                pass

    # add counts and export tables
    agg_all = p_val_count(agg_all)
    agg_rcp26 = p_val_count(agg_rcp26)
    agg_rcp45 = p_val_count(agg_rcp45)
    agg_rcp60 = p_val_count(agg_rcp60)
    agg_rcp85 = p_val_count(agg_rcp85)

    if parametric:
        agg_all.to_csv('single_scenario_data/parametric/' + objective + '/aggregate_all.csv')
        agg_rcp26.to_csv('single_scenario_data/parametric/' + objective + '/aggregate_rcp26.csv')
        agg_rcp45.to_csv('single_scenario_data/parametric/' + objective + '/aggregate_rcp45.csv')
        agg_rcp60.to_csv('single_scenario_data/parametric/' + objective + '/aggregate_rcp60.csv')
        agg_rcp85.to_csv('single_scenario_data/parametric/' + objective + '/aggregate_rcp85.csv')
    else:
        agg_all.to_csv('single_scenario_data/nonparametric/' + objective + '/aggregate_all.csv')
        agg_rcp26.to_csv('single_scenario_data/nonparametric/' + objective + '/aggregate_rcp26.csv')
        agg_rcp45.to_csv('single_scenario_data/nonparametric/' + objective + '/aggregate_rcp45.csv')
        agg_rcp60.to_csv('single_scenario_data/nonparametric/' + objective + '/aggregate_rcp60.csv')
        agg_rcp85.to_csv('single_scenario_data/nonparametric/' + objective + '/aggregate_rcp85.csv')

    # plot total counts (all scenarios)
    agg_all.loc['2000-10-1':'2098-10-1', 'count'].plot()
    plt.ylabel('significance counts')
    # plt.show()
    if parametric:
        plt.savefig('single_scenario_data/parametric/' + objective + '/total_counts.png')
    else:
        plt.savefig('single_scenario_data/nonparametric/' + objective + '/total_counts.png')
    plt.clf()

    # make plot of final results
    df = pd.read_csv('empty/datetime.csv', index_col=0,
                     parse_dates=True)

    # store rcp-specific counts relative to number of models in group
    df['rcp26'] = agg_rcp26['count'] / (len(list(agg_rcp26)) - 1)
    df['rcp45'] = agg_rcp45['count'] / (len(list(agg_rcp45)) - 1)
    df['rcp60'] = agg_rcp60['count'] / (len(list(agg_rcp60)) - 1)
    df['rcp85'] = agg_rcp85['count'] / (len(list(agg_rcp85)) - 1)

    df = df['2000-10-1':'2098-10-1']

    df.plot()
    plt.ylabel('relative significance counts')
    # plt.show()
    if parametric:
        plt.savefig('single_scenario_data/parametric/' + objective + '/rcp_relative_counts.png')
    else:
        plt.savefig('single_scenario_data/nonparametric/' + objective + '/rcp_relative_counts.png')
    plt.clf()

    return


export_plt('reliability', parametric=True)
export_plt('reliability', parametric=False)
export_plt('flood', parametric=True)
export_plt('flood', parametric=False)
