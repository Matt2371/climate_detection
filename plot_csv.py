import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

# supress warnings
import warnings
warnings.filterwarnings('ignore')

### Plot additional figures summarizing csv results from plot_results.py
## Stored in /additional_materials/ folders

## Shared parameters:
## objective: ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs'], str
## win_size: moving average window size applied, int
## alt = ['two-sided', 'less', 'greater']

## For multi-scenarios: plot distribution of rel_counts at specified year
# Parameters: sort_by ('byrcp'/'bygcm'/'bylulc'), year (year of interest)
def last_yr_hist(objective, alt, win_size, sort_by, year):
    # read csv of interest
    data = pd.read_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/'
                       + alt + '_multi_' + sort_by + '_win' +
                       str(win_size) + '.csv', index_col=0, parse_dates=True)
    # plot year of interest, save figure
    data.loc[str(year) + '-10-01', :].plot.hist(bins=10)
    plt.ylabel('frequency')
    plt.xlabel('relative counts')
    plt.title('distribution of rel counts ' + sort_by + ' in ' + str(year))
    plt.savefig(
        'significance_results/nonparametric/' + objective + '/additional_materials/' + alt + '_multi_' + sort_by + '_win' +
        str(win_size) + '_' + str(year) + '.png')
    plt.clf()

    return


## For single-scenarios: get table of stats (median, std) of first detection year distribution for each sorting type
def single_stats(objective, alt, win_size, sort_by, pre_whitening):
    # read csv of interest
    if pre_whitening:
        filename = f'significance_results/nonparametric/{objective}/{str(win_size)}_year_MA/{alt}_single_{sort_by}_win{str(win_size)}_pw.csv'
    else:
        filename = f'significance_results/nonparametric/{objective}/{str(win_size)}_year_MA/{alt}_single_{sort_by}_win{str(win_size)}.csv'

    data = pd.read_csv(filename, index_col=0)
    # make a list of columns (names of gcm/rcp/lulc scenario names)
    columns = list(data)
    # initialize empty list to store statistics
    median_list = []
    stdv_list = []
    # number of models with detection year
    sample_size = []

    # collect data for each column
    for col in columns:
        # iterate over rows: 'row' stores detection years
        # initialize sample size counter
        i = 0
        for row in data[[col]].itertuples(index=False):
            if not np.isnan(row[0]):
                i += 1
        # add statistics
        sample_size.append(i)
        median_list.append(data[col].median())
        stdv_list.append(data[col].std())

    df_data = {sort_by: columns, 'Median': median_list, 'Standard_deviation': stdv_list, 'Sample_size': sample_size}
    df = pd.DataFrame(df_data)

    if pre_whitening:
        save_dir = f'significance_results/nonparametric/{objective}/additional_materials/{alt}_single_{sort_by}_win{str(win_size)}_stats_pw.csv'
    else:
        save_dir = f'significance_results/nonparametric/{objective}/additional_materials/{alt}_single_{sort_by}_win{str(win_size)}_stats.csv'
        
    df.to_csv(save_dir)

    return df


# Single scenario: Plot first detection year against objective value (moving average) at DETECTION YEAR
# NOTE: SKIPS NO DETECT SCENARIOS
def detect_vs_obj(objective, alt, win_size):
    # initiate empty lists to store data
    model_list = []
    detection_year = []
    objective_values = []

    # load csv with all first detection years
    df_years = pd.read_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                           alt + '_single_total_' + 'win' + str(win_size) + '.csv', index_col=0)
    # iterate over models, row stores model name and first detection year
    for row in tqdm(df_years.itertuples(index=False), desc='Building df...'):
        # SKIP no detect models
        if np.isnan(row[1]):
            continue
        else:
            model_list.append(row[0])
            detection_year.append(row[1])

            # extract gcm/rcp/lulc from model name
            gcm = row[0].split('_')[0]
            rcp = row[0].split('_')[1]
            try:
                lulc = row[0].split('_')[2] + '_' + row[0].split('_')[3]
            except IndexError:
                lulc = row[0].split('_')[2]

            # read model objective value (trailing average)
            scenario = gcm + '_' + rcp + '_r1i1p1'
            df_obj = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0, parse_dates=True)
            objective_values.append(
                df_obj.loc[str(int(row[1]) - win_size + 1) + '-10-1':str(int(row[1])) + '-10-1', objective].mean())
            # print(df_obj.loc[str(int(row[1])-win_size)+'-10-1':str(int(row[1]))+'-10-1', objective])

    # build data frame and export plot
    data = {'Model': model_list, 'Detection_year': detection_year, objective: objective_values}
    output_df = pd.DataFrame(data)
    output_df.plot.scatter(x='Detection_year', y=objective)
    plt.xlabel('detection year')
    # plt.title('Detection Year vs Objective Severity (30-yr MA)')
    plt.savefig(
        'significance_results/nonparametric/' + objective + '/additional_materials/' + 'detectvsobj_' + alt + '_single_'
        + 'win' + str(win_size) + '.png')
    plt.clf()

    return output_df


# Single scenario: Plot first detection year against objective value (moving average) at END OF PROJECTION (2098)
# INCLUDES DISTRIBUTION OF NO DETECT SCENAIORS' SEVERITY (BOXPLOT)
def detect_vs_end_obj(objective, alt, win_size):
    # initiate empty lists to store data WITH DETECTION
    model_list = []
    detection_year = []
    objective_values = []

    # initiate empty lists to store data from NO DETECT scenarios
    model_list_nd = []
    objective_values_nd = []

    # load csv with all first detection years
    df_years = pd.read_csv('significance_results/nonparametric/' + objective + '/' + str(win_size) + '_year_MA/' +
                           alt + '_single_total_' + 'win' + str(win_size) + '.csv', index_col=0)
    # iterate over models, row stores model name and first detection year
    for row in tqdm(df_years.itertuples(index=False), desc='Building df...'):
        # handle no detect models
        if np.isnan(row[1]):
            model_list_nd.append(row[0])

            # extract gcm/rcp/lulc from model name
            gcm = row[0].split('_')[0]
            rcp = row[0].split('_')[1]
            try:
                lulc = row[0].split('_')[2] + '_' + row[0].split('_')[3]
            except IndexError:
                lulc = row[0].split('_')[2]

            # read model objective value at 2098 (trailing average)
            scenario = gcm + '_' + rcp + '_r1i1p1'
            df_obj = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0, parse_dates=True)
            objective_values_nd.append(df_obj.loc[str(2098 - win_size + 1) + '-10-1':'2098-10-1', objective].mean())

        else:
            model_list.append(row[0])
            detection_year.append(row[1])

            # extract gcm/rcp/lulc from model name
            gcm = row[0].split('_')[0]
            rcp = row[0].split('_')[1]
            try:
                lulc = row[0].split('_')[2] + '_' + row[0].split('_')[3]
            except IndexError:
                lulc = row[0].split('_')[2]

            # read model objective value at 2098 (trailing average)
            scenario = gcm + '_' + rcp + '_r1i1p1'
            df_obj = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0, parse_dates=True)
            objective_values.append(df_obj.loc[str(2098 - win_size + 1) + '-10-1':'2098-10-1', objective].mean())

    # build data frames and export plot ('ND', "no detection", stores severity of no detect scenarios)
    data = {'Model': model_list, 'Detection_year': detection_year, objective: objective_values}
    data_nd = {'Model': model_list_nd, 'ND': objective_values_nd}
    output_df = pd.DataFrame(data)
    output_nd_df = pd.DataFrame(data_nd)

    # create scatter (right) and boxplot of values for scenarios with no detection (left)
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, wspace=0, width_ratios=[10, 1])
    axs = gs.subplots(sharey=True)

    output_df.plot.scatter(x='Detection_year', y=objective, ax=axs[0])
    boxplot = output_nd_df.boxplot(column='ND', grid=False, return_type='dict', ax=axs[1], widths=0.7)

    # customize plots
    # see boxplot customization options:
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html#matplotlib.pyplot.boxplot
    axs[1].spines['left'].set_visible(False)
    axs[1].tick_params(left=False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_xlabel('detection year')
    boxplot['fliers'][0].set(markersize=4, mec='black')
    boxplot['boxes'][0].set(linewidth=1, color='black')
    boxplot['medians'][0].set(linewidth=1, color='black')
    for cap in boxplot['caps']:
        cap.set(linewidth=1, color='black')
    for whisker in boxplot['whiskers']:
        whisker.set(linewidth=1, color='black')

    plt.savefig(
        'significance_results/nonparametric/' + objective + '/additional_materials/' + 'detect_2098obj_' + alt + '_single_'
        + 'win' + str(win_size) + '.png')
    plt.clf()

    return output_df


# Plot distribution of historical objectives vs objectives from observed inflows
def historical_dist(objective):
    # Store scenario names
    gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
                'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
                'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
                'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
                'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']
    rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    lulc_list = pd.read_csv('lulc_scenario_names.csv').name.to_list()

    # store model historical results
    model_historical = []
    model_scenarios = []

    # store stats for table
    median = []
    stdev = []
    observed = []

    # check every gcm/rcp/lulc combination
    for gcm in tqdm(gcm_list, desc='Building historical distribution: '):
        for rcp in rcp_list:
            for lulc in lulc_list:
                try:
                    # read all scenarios, plot rolling average
                    scenario = gcm + '_' + rcp + '_r1i1p1'
                    data_df = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0,
                                          parse_dates=True)
                    model_historical.append(data_df.loc['1997-10-1':'2020-10-1', objective].mean())
                    model_scenarios.append(gcm + '_' + rcp + '_' + lulc)

                except FileNotFoundError:
                    pass
    historical_data = {'Model': model_scenarios, objective: model_historical}
    historical_df = pd.DataFrame(historical_data)

    # get objective value from observed data
    observed_df = pd.read_csv('obj_historical.csv', index_col=0,
                              parse_dates=True)
    x_value = observed_df.loc['1997-10-1':'2020-10-1', objective].mean()

    # add values to table and save
    median.append(historical_df[objective].median())
    stdev.append(historical_df[objective].std())
    observed.append(x_value)
    summary_data = {'median': median, 'stdev': stdev, 'observed': observed}
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('significance_results/nonparametric/' + objective +
                      '/additional_materials/model_vs_obs_historical_stats.csv')

    # plot and save results
    historical_df[objective].plot.hist(bins=15)
    plt.axvline(x=x_value, color='black', linestyle='--')
    plt.xlabel(objective)
    plt.legend(['observed', 'modeled'])
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.savefig('significance_results/nonparametric/' + objective + '/additional_materials/model_vs_obs_historical.png')
    plt.clf()

    return


# export results for each objective
obj_list = ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf']
sortby_list = ['bygcm', 'byrcp', 'bylulc']
year = 2098
win_size = 30

for objective in obj_list:
    if objective in ['Rel_NOD_%', 'Rel_SOD_%']:
        alt = 'less'
    if objective == 'Upstream_Flood_Volume_taf':
        alt = 'greater'
    for sort in sortby_list:
        last_yr_hist(objective, alt, win_size, sort, year)

        for pre_whitening in [True, False]:
            single_stats(objective, alt, win_size, sort, pre_whitening)

    detect_vs_obj(objective, alt, win_size)
    detect_vs_end_obj(objective, alt, win_size)
    historical_dist(objective)

historical_dist('Delta_Peak_Inflow_cfs')

## tests
# historical_dist('Delta_Peak_Inflow_cfs')
# detect_vs_end_obj('Rel_NOD_%', 'less', 30)
# detect_vs_obj('Upstream_Flood_Volume_taf', 'greater', 30)
# historical_dist('Rel_NOD_%')
