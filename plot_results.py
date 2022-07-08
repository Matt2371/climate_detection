import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import significance_detection_v4 as sd
from tqdm import tqdm

### Exports data/results from analysis functions defined in significance_detection_v4.py (sd)

## Shared parameters:
## objective: ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs'], str
## win_size: moving average window size applied, int
## alt = ['two-sided', 'less', 'greater']


# store names of gcm/rcp/lulc combinations
gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
            'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
            'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
            'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
            'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']

rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
lulc_names = pd.read_csv('lulc_scenario_names.csv').name.to_list()


## Plot total relative counts (multiple scenario) per year, save all p_vals as csv
# agg_all=df of all p_vals (from significance_detection.py, export_agg()). NEED TO ADD COUNTS (sd.p_val_count(df))
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_multi_total(agg_all, objective, alt, win_size):
    ## Get p-vals, add counts
    # agg_all = sd.p_val_count(sd.export_agg(objective, alt=alt, win_size=win_size))
    agg_all['rel_count'] = agg_all['count'] / (len(list(agg_all)) - 1)

    # Plot total relative counts, save agg_all df
    agg_all.loc['2000-10-1':'2098-10-1', 'rel_count'].plot()
    plt.ylabel('relative significance counts')
    plt.savefig('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                alt + '_multi_total_win' + str(win_size) + '.png')
    plt.clf()
    agg_all.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                   alt + '_pvals_win' + str(win_size) + '.csv')

    return


## Plot total relative counts (multiple scenario) separated by rcp, per year
# agg_all=df of all p_vals (from significance_detection.py, export_agg()). DOES NOT MATTER IF COUNTS ARE NOT ADDED
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_multi_rcp(agg_all, objective, alt, win_size):
    ## Get p-vals
    # agg_all = sd.export_agg(objective, alt=alt, win_size=win_size)

    ## Separate by RCP
    # Add empty df with dates to add counts to, sorted by rcp
    counts_rcp = pd.read_csv('empty/datetime.csv', index_col=0,
                             parse_dates=True)
    # do one rcp at a time
    for rcp in rcp_list:
        # temporary df to store values for one rcp
        temp = pd.read_csv('empty/datetime.csv', index_col=0,
                           parse_dates=True)

        # check one col at a time for rcp, makes sub df with rcp's of only one type
        for col in list(agg_all):
            if rcp in col:
                temp[col] = agg_all[col]
        # add (relative) counts from temp df and save
        temp = sd.p_val_count(temp)
        counts_rcp[rcp + '_rel_counts'] = temp['count'] / (len(list(temp)) - 1)

    ## export results
    counts_rcp['2000-10-1':'2098-10-1'].plot()
    plt.ylabel('relative significance counts')
    plt.legend(rcp_list)
    plt.savefig('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                alt + '_multi_byrcp_win' + str(win_size) + '.png')
    plt.clf()
    counts_rcp.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                      alt + '_multi_byrcp_win' + str(win_size) + '.csv')

    return


## Plot total relative counts (multiple scenario) separated by gcm, per year
# agg_all=df of all p_vals (from significance_detection.py, export_agg()). DOES NOT MATTER IF COUNTS ARE NOT ADDED
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_multi_gcm(agg_all, objective, alt, win_size):
    ## Get p-vals
    # agg_all = sd.export_agg(objective, alt=alt, win_size=win_size)

    ## Separate by GCM
    # Add empty df with dates to add counts to, sorted by gcm
    counts_gcm = pd.read_csv('empty/datetime.csv', index_col=0,
                             parse_dates=True)
    # do one gcm at a time
    for gcm in gcm_list:
        # temporary df to store values for one gcm
        temp = pd.read_csv('empty/datetime.csv', index_col=0,
                           parse_dates=True)

        # check one col at a time for gcm, makes sub df with gcm's of only one type
        for col in list(agg_all):
            if gcm in col:
                temp[col] = agg_all[col]
        # add (relative) counts from temp df and save
        temp = sd.p_val_count(temp)
        counts_gcm[gcm + '_rel_counts'] = temp['count'] / (len(list(temp)) - 1)

    ## export results
    counts_gcm.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                      alt + '_multi_bygcm_win' + str(win_size) + '.csv')

    return


## Plot total relative counts (multiple scenario) separated by lulc, per year
# agg_all=df of all p_vals (from significance_detection.py, export_agg())
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_multi_lulc(agg_all, objective, alt, win_size):
    ## Get p-vals
    # agg_all = sd.export_agg(objective, alt=alt, win_size=win_size)

    ## Separate by lulc
    # Add empty df with dates to add counts to, sorted by lulc
    counts_lulc = pd.read_csv('empty/datetime.csv', index_col=0,
                              parse_dates=True)
    # do one lulc at a time
    for lulc in lulc_names:
        # temporary df to store values for one lulc
        temp = pd.read_csv('empty/datetime.csv', index_col=0,
                           parse_dates=True)

        # check one col at a time for lulc, makes sub df with lulc's of only one type
        for col in list(agg_all):
            if lulc in col:
                temp[col] = agg_all[col]
        # add (relative) counts from temp df and save
        temp = sd.p_val_count(temp)
        counts_lulc[lulc + '_rel_counts'] = temp['count'] / (len(list(temp)) - 1)

    # export results
    counts_lulc.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                       alt + '_multi_bylulc_win' + str(win_size) + '.csv')

    return


## plot distribution of years of first significance (single scenario), export years as csv
# agg_all=df of all p_vals (from significance_detection.py, export_agg()). NEED TO REMOVE COUNTS (MODELS ONLY)
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_single_total(agg_all, objective, alt, win_size):
    ## Get p-vals
    # agg_all = sd.export_agg(objective, alt=alt, win_size=win_size)

    years = sd.first_significance(agg_all)
    years.plot.hist(bins=20, legend=False)
    plt.xlabel('detection year')

    ## export results
    plt.savefig('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                alt + '_single_hist_win' + str(win_size) + '.png')
    plt.clf()
    years.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                 alt + '_single_total_win' + str(win_size) + '.csv')
    return


## plot distribution of years of first significance (single scenario), sorted by rcp
# agg_all=df of all p_vals (from significance_detection.py, export_agg()). NEED TO REMOVE COUNTS (MODELS ONLY)
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_single_rcp(agg_all, objective, alt, win_size):
    ## Get p-vals
    # agg_all = sd.export_agg(objective, alt=alt, win_size=win_size)

    ## Separate by RCP
    # Add empty df to add years of first detection, sorted by rcp
    years_rcp = pd.DataFrame(index=range(len(gcm_list) * len(lulc_names)), columns=rcp_list)
    # do one rcp at a time
    for rcp in rcp_list:
        # temporary df to store values for one rcp, reset for every rcp
        temp = pd.read_csv('empty/datetime.csv', index_col=0,
                           parse_dates=True)

        # make total list of cols in agg_all df, check one col at a time for rcp name
        for col in list(agg_all):
            if rcp in col:
                temp[col] = agg_all[col]

        # temp filters all p-vals (from all relevant scenarios) for a single rcp after completing loop through cols
        # find years of first significance from temp
        years_temp = sd.first_significance(temp)
        # add sample of years to df
        years_rcp[rcp] = years_temp['Year']

    ## export/plot results
    # plot stacked histogram
    years_rcp.plot.hist(stacked=True, bins=20)
    plt.xlabel('Year')
    plt.savefig('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                alt + '_single_hist_byrcp_win' + str(win_size) + '.png')
    plt.clf()
    # plot subplots sorted by rcp
    years_rcp.hist(bins=10, grid=False)
    plt.savefig('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                alt + '_single_subplots_byrcp_win' + str(win_size) + '.png')
    plt.clf()
    years_rcp.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                     alt + '_single_byrcp_win' + str(win_size) + '.csv')
    return


## plot distribution of years of first significance (single scenario), sorted by gcm
# agg_all=df of all p_vals (from significance_detection.py, export_agg()). NEED TO REMOVE COUNTS (MODELS ONLY)
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_single_gcm(agg_all, objective, alt, win_size):
    ## Get all p-vals (for all scenarios)
    # agg_all = sd.export_agg(objective, alt=alt, win_size=win_size)

    ## Separate by GCM
    # Add empty df to add years of first detection, sorted by gcm
    years_gcm = pd.DataFrame(index=range(len(rcp_list) * len(lulc_names)), columns=gcm_list)
    # do one gcm at a time
    for gcm in gcm_list:
        # temporary df to store values for one gcm, reset for every gcm
        temp = pd.read_csv('empty/datetime.csv', index_col=0,
                           parse_dates=True)

        # make total list of cols in agg_all df, check one col at a time for gcm name
        for col in list(agg_all):
            if gcm in col:
                temp[col] = agg_all[col]

        # temp filters all p-vals (from all relevant scenarios) for a single gcm after completing loop through cols
        # find years of first significance from temp
        years_temp = sd.first_significance(temp)
        # add sample of years to df
        years_gcm[gcm] = years_temp['Year']

    ## export results
    years_gcm.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                     alt + '_single_bygcm_win' + str(win_size) + '.csv')
    return


## plot distribution of years of first significance (single scenario), sorted by gcm
# agg_all=df of all p_vals (from significance_detection.py, export_agg()). NEED TO REMOVE COUNTS (MODELS ONLY)
# make sure agg_all is consistent with other parameters (objective, alt, win_size)
def plot_single_lulc(agg_all, objective, alt, win_size):
    ## Get all p-vals (for all scenarios)
    # agg_all = sd.export_agg(objective, alt=alt, win_size=win_size)

    ## Separate by LULC
    # Add empty df to add years of first detection, sorted by lulc
    years_lulc = pd.DataFrame(index=range(len(rcp_list) * len(gcm_list)), columns=lulc_names)
    # do one lulc at a time
    for lulc in lulc_names:
        # temporary df to store values for one lulc, reset for every lulc
        temp = pd.read_csv('empty/datetime.csv', index_col=0,
                           parse_dates=True)

        # make total list of cols in agg_all df, check one col at a time for lulc name
        for col in list(agg_all):
            if lulc in col:
                temp[col] = agg_all[col]

        # temp filters all p-vals (from all relevant scenarios) for a single lulc after completing loop through cols
        # find years of first significance from temp
        years_temp = sd.first_significance(temp)
        # add sample of years to df
        years_lulc[lulc] = years_temp['Year']

    ## plot/export results
    years_lulc.to_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                      alt + '_single_bylulc_win' + str(win_size) + '.csv')
    return


def main():
    # list of objectives
    obj_list = ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs']
    # set window size
    win_size = 30

    # export results for each objective
    for objective in tqdm(obj_list, desc='Exporting Results'):
        if objective in ['Rel_NOD_%', 'Rel_SOD_%']:
            alt = 'less'
            # get p-vals and counts
            agg_all = sd.p_val_count(sd.export_agg(objective, alt=alt, win_size=win_size))

            plot_multi_total(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_multi_rcp(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_multi_gcm(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_multi_lulc(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)

            # delete count and rel_count columns for single scenario analysis
            agg_all = agg_all.drop(['count', 'rel_count'], axis='columns')

            plot_single_total(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_single_rcp(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_single_gcm(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_single_lulc(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)

        if objective == 'Upstream_Flood_Volume_taf':
            alt = 'greater'
            agg_all = sd.p_val_count(sd.export_agg(objective, alt=alt, win_size=win_size))

            plot_multi_total(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_multi_rcp(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_multi_gcm(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_multi_lulc(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)

            # delete count and rel_count columns for single scenario analysis
            agg_all = agg_all.drop(['count', 'rel_count'], axis='columns')

            plot_single_total(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_single_rcp(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_single_gcm(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
            plot_single_lulc(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    return

    if __name__ == "__main__":
        main()

    # if objective == 'Delta_Peak_Inflow_cfs':
    #     alt = 'two-sided'
    #     agg_all = sd.p_val_count(sd.export_agg(objective, alt=alt, win_size=win_size))
    #     plot_multi_total(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    #     plot_multi_rcp(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    #     plot_multi_gcm(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    #     plot_multi_lulc(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    #
    #     # delete count and rel_count columns for single scenario analysis
    #     agg_all = agg_all.drop(['count', 'rel_count'], axis='columns')
    #
    #     plot_single_total(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    #     plot_single_rcp(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    #     plot_single_gcm(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
    #     plot_single_lulc(agg_all=agg_all, objective=objective, alt=alt, win_size=win_size)
