import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


# take rolling averages, run statistical significance tests against historical
# parameters: gcm, rcp, objective (reliability/flood), parametric (t-test or MWU),
# alt ('greater/less/two-sided' hypothesis compared to historical), win_size(size of MA window)
# returns: dataframe with original data AND p-values
def rolling_significance(gcm, rcp, objective, parametric=False, alt='two-sided', win_size=30):
    # load dataframe and isolate objective
    scenario = gcm + '_' + rcp + '_r1i1p1'
    df = pd.read_csv('data/%s-results.csv' % scenario, index_col=0, parse_dates=True)
    df = df[[objective]]

    # set historical df's
    his_df = df['1951-10-01':'2000-10-01'].copy()
    # set projection df based on window size
    lower_yr = str(2000-win_size+1)
    proj_df = df[lower_yr+'-10-01':'2098-10-01'].copy()

    # get rolling samples (future), delete anything before year 2000
    proj_df['rolling'] = [window.to_list() for window in proj_df.loc[:, objective].rolling(win_size)]
    proj_df.loc[lower_yr+'-10-01':'1999-10-01', 'rolling'] = float("NaN")

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


# store names of gcm/rcp combinations
gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
            'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
            'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
            'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
            'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']

rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']


# returns tables with p values aggregated for every scenario (gcm/rcp combination)
# parameters: objective, parametric, alt, win_size as defined by rolling_significance()
def export_agg(objective, parametric=False, alt='two-sided', win_size=30):
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
                p_vals = rolling_significance(gcm, rcp, objective, parametric, alt, win_size)[objective + '_p-value']
                agg_all[gcm + '_' + rcp] = p_vals

                # add p-vals to rcp specific df's as well
                # e.g. conditions to isolate
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

    return agg_all, agg_rcp26, agg_rcp45, agg_rcp60, agg_rcp85


# for aggregate df's (see fun below below), add "count" column that keeps tract of p<0.05 for each datetime (multiple
# scenario) returns: inputted dataframe with added "count" column
def p_val_count(df):
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


# returns: year of first significance for each model scenario (single scenario)
# parameter: input df with p values
def first_significance(df):
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
        # 'row' stores datetime, col p value
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
    output['Year'] = output['Date'].dt.year
    output = output.drop(['Date'], axis=1)

    # print(output)
    return output


# # create empty data frame
# df = pd.DataFrame()
# df['rcp26'] = first_significance(export_agg('flood', alt='greater')[1])['Year']
# df['rcp45'] = first_significance(export_agg('flood', alt='greater')[2])['Year']
# df['rcp60'] = first_significance(export_agg('flood', alt='greater')[3])['Year']
# df['rcp85'] = first_significance(export_agg('flood', alt='greater')[4])['Year']
#
# df.plot.hist(stacked=True, bins=20)
# plt.xlabel('Year')
# plt.savefig('significance_data/nonparametric/reliability/single_stacked_his1.png')
# plt.clf()
#
# df.hist(bins=10, grid=False)
# plt.savefig('significance_data/nonparametric/reliability/single_rcp_hists1.png')

# agg_all = p_val_count(export_agg('flood', parametric=False, alt='greater')[0])
# agg_all['count'] = agg_all['count'] / (len(list(agg_all)) - 1)
# agg_all.loc['2000-10-1':'2098-10-1', 'count'].plot()
# plt.ylabel('relative significance counts')
# plt.savefig('significance_data/parametric/flood/one_side_total_counts.png')
#
# agg_all.to_csv('one_side_agg_all.csv')