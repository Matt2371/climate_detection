import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from plot_ensemble import ensemble
import significance_detection_v4 as sd


### Create figures to be used in journal article

## Ensemble subplots for water supply reliability (SOD) and upstream flood volume and add scatterplot for observations
def plot_ensemble():
    df_rel = ensemble('Rel_SOD_%')
    df_flood = ensemble('Upstream_Flood_Volume_taf')
    # find historical range (max/min)
    rel_min = df_rel['1981-10-1':'2000-10-1'].min().min()
    rel_max = df_rel['1981-10-1':'2000-10-1'].max().max()
    flood_min = df_flood['1981-10-1':'2000-10-1'].min().min()
    flood_max = df_flood['1981-10-1':'2000-10-1'].max().max()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 4])

    # add scatterplot of observed values (20-year average)
    obs = pd.read_csv('obj_historical.csv', index_col=0, parse_dates=True)
    obs.loc['2020-10-1', :] = obs.loc['2000-10-1':'2020-10-1'].mean(axis=0)
    obs['datetime'] = obs.index

    axes[0].plot(obs.loc['2020-10-1', 'datetime'], obs.loc['2020-10-1', 'Rel_SOD_%'], c='red', marker='o',
                 markerfacecolor='red', markersize=5)
    axes[1].plot(obs.loc['2020-10-1', 'datetime'], obs.loc['2020-10-1', 'Upstream_Flood_Volume_taf'],
                 c='red', marker='o', markerfacecolor='red', markersize=5)

    # plot projections
    df_rel['2000-10-1':'2098-10-1'].plot(legend=False, c='lightgray', ax=axes[0], zorder=0)
    df_flood['2000-10-1':'2098-10-1'].plot(legend=False, c='lightgray', ax=axes[1], zorder=0)

    # plot mean
    df_rel.loc['2000-10-1':'2098-10-1', 'mean'].plot(c='steelblue', ax=axes[0], zorder=10)
    df_flood.loc['2000-10-1':'2098-10-1', 'mean'].plot(c='steelblue', ax=axes[1], zorder=10)

    # plot historical ranges
    axes[0].fill_between(df_rel['2000-10-1':'2098-10-1'].index, rel_min, rel_max,
                         facecolor='green', alpha=0.2, zorder=5)
    axes[1].fill_between(df_flood['2000-10-1':'2098-10-1'].index, flood_min, flood_max,
                         facecolor='green', alpha=0.2, zorder=5)

    axes[0].set_ylabel('reliability')
    axes[0].set_title('Water supply reliability projections, 30 yr MA')
    axes[1].set_ylabel('flood volume (TAF)')
    axes[1].set_title('Upstream flood volume projections, 30 yr MA')

    # create custom legend
    legend_elements = [Line2D([0], [0], color='lightgray', label='projections'),
                       Line2D([0], [0], color='steelblue', label='projections mean'),
                       Patch(facecolor='green', alpha=0.5, label='historical range'),
                       Line2D([0], [0], marker='o', color='w', label='observed value (20-yr avg)',
                              markerfacecolor='red', markersize=5)]
    axes[0].legend(handles=legend_elements)
    axes[1].legend(handles=legend_elements)

    plt.tight_layout()
    # plt.show()
    plt.savefig('significance_results/article_figures/ensemble_plot.png')
    plt.clf()

    return


## Plot distribution for first detection years for water supply reliability (SOD) and upstream flood volume
def plot_single_total(win_size=30):
    # Load first detection years from csv's
    years_rel = pd.read_csv('significance_results/nonparametric/' + 'Rel_SOD_%' + '/' + str(win_size) + '_year_MA/' +
                            'less' + '_single_total_win' + str(win_size) + '.csv', index_col=0)['Year']
    years_flood = pd.read_csv('significance_results/nonparametric/' + 'Upstream_Flood_Volume_taf' + '/' + str(win_size)
                              + '_year_MA/' + 'greater' + '_single_total_win' + str(win_size) + '.csv', index_col=0)[
        'Year']

    # create subpolots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 4], sharey=True)
    years_rel.plot.hist(bins=20, legend=False, ax=axes[0])
    years_flood.plot.hist(bins=20, legend=False, ax=axes[1])
    axes[0].set_xlabel('detection year')
    axes[0].set_title('First detection years for water supply reliability')
    axes[1].set_xlabel('detection year')
    axes[1].set_title('First detection years for upstream flood volume')

    ## export results
    plt.tight_layout()
    plt.savefig('significance_results/article_figures/first_detection_years.png')
    plt.clf()
    return


## Plot stats of first detection years (sorted by gcm/rcp/lulc)
# objective = 'Rel_SOD_%' or 'Upstream_Flood_Volume_taf'
def plot_single_sorted(objective, win_size=30):
    # set alternative and figure title based on objective
    if objective == 'Rel_SOD_%':
        alt = 'less'
        name = 'water supply reliability'
    else:
        alt = 'greater'
        name = 'upstream flood volume'

    # create subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=[10, 8])
    fig.suptitle('Descriptive statistics for first detection years (' + name + ')', size='xx-large')

    # plot gcm stats
    gcm_stat = pd.read_csv('significance_results/nonparametric/' + objective + '/additional_materials/' + alt +
                           '_single_' + 'bygcm' + '_win' + str(win_size) + '_stats.csv')
    gcm_stat = gcm_stat[['Median', 'Standard_deviation', 'Sample_size']]
    gcm_stat['Median'].plot.hist(ax=axes[0, 0])
    axes[0, 0].set_xlabel('detection year')
    axes[0, 0].set_title('Median sorted by GCM')
    gcm_stat['Standard_deviation'].plot.hist(ax=axes[0, 1])
    axes[0, 1].set_xlabel('stdev detection year')
    axes[0, 1].set_title('Stdev sorted by GCM')
    gcm_stat['Sample_size'].plot.hist(ax=axes[0, 2])
    axes[0, 2].set_xlabel('sample size')
    axes[0, 2].set_title('Sample sizes sorted by GCM')

    # plot lulc stats
    lulc_stat = pd.read_csv(
        'significance_results/nonparametric/' + objective + '/additional_materials/' + alt + '_single_' + 'bylulc' +
        '_win' + str(win_size) + '_stats.csv')
    lulc_stat = lulc_stat[['Median', 'Standard_deviation', 'Sample_size']]
    lulc_stat['Median'].plot.hist(ax=axes[1, 0])
    axes[1, 0].set_xlabel('detection year')
    axes[1, 0].set_title('Median sorted by LULC')
    lulc_stat['Standard_deviation'].plot.hist(ax=axes[1, 1])
    axes[1, 1].set_xlabel('stdev detection year')
    axes[1, 1].set_title('Stdev sorted by LULC')
    lulc_stat['Sample_size'].plot.hist(ax=axes[1, 2])
    axes[1, 2].set_xlabel('sample size')
    axes[1, 2].set_title('Sample sizes sorted by LULC')

    # plot rcp stats
    rcp_stat = pd.read_csv('significance_results/nonparametric/' + objective + '/additional_materials/' + alt +
                           '_single_' + 'byrcp' + '_win' + str(win_size) + '_stats.csv')

    axes[2, 0].bar(rcp_stat['byrcp'], rcp_stat['Median'], width=0.4)
    axes[2, 0].set_ylim(bottom=2000, top=2075)
    axes[2, 0].set_ylabel('Detection year')
    axes[2, 0].set_title('Median sorted by RCP')
    axes[2, 1].bar(rcp_stat['byrcp'], rcp_stat['Standard_deviation'], width=0.4)
    axes[2, 1].set_ylabel('Stdev detection year')
    axes[2, 1].set_title('Stdev sorted by RCP')
    axes[2, 2].bar(rcp_stat['byrcp'], rcp_stat['Sample_size'], width=0.4)
    axes[2, 2].set_ylabel('Sample size')
    axes[2, 2].set_title('Sample sizes sorted by RCP')

    plt.tight_layout()
    plt.savefig('significance_results/article_figures/first_detection_stats_' + objective + '.png')
    plt.clf()

    return


## Plot detection rates for water supply reliability (SOD) and upstream flood volume
def plot_multi_total(win_size=30):
    # Load p-val data (rel counts is included as a column)
    agg_all_rel = pd.read_csv('significance_results/nonparametric/' + 'Rel_SOD_%' + '/' + str(win_size) + '_year_MA/' +
                              'less_pvals_win30.csv', index_col=0, parse_dates=True)
    agg_all_flood = pd.read_csv('significance_results/nonparametric/' + 'Upstream_Flood_Volume_taf' + '/' +
                                str(win_size) + '_year_MA/' + 'greater_pvals_win30.csv', index_col=0, parse_dates=True)

    # Plot total relative counts
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=[10, 4.5])
    agg_all_rel.loc['2000-10-1':'2098-10-1', 'rel_count'].plot(ax=axes[0])
    agg_all_flood.loc['2000-10-1':'2098-10-1', 'rel_count'].plot(ax=axes[1])

    axes[0].set_ylabel('detection rate')
    axes[0].set_title('Water supply reliability')
    axes[1].set_ylabel('detection rate')
    axes[1].set_title('Upstream flood volume')
    fig.suptitle('Fraction of scenarios with detection')

    plt.tight_layout()
    # plt.show()
    plt.savefig('significance_results/article_figures/detection_rate.png')
    plt.clf()

    return


## Create histograms for detection rates by gcm/rcp/lulc at the end of simulation (2098)
def plot_multi_sorted(win_size=30):
    # load detection rates for water supply reliability sorted by gcm/rcp/lulc
    df_gcm_rel = pd.read_csv('significance_results/nonparametric/' + 'Rel_SOD_%' + '/' + str(win_size) + '_year_MA/'
                             + 'less' + '_multi_' + 'bygcm' + '_win' +
                             str(win_size) + '.csv', index_col=0, parse_dates=True)
    df_rcp_rel = pd.read_csv('significance_results/nonparametric/' + 'Rel_SOD_%' + '/' + str(win_size) + '_year_MA/'
                             + 'less' + '_multi_' + 'byrcp' + '_win' +
                             str(win_size) + '.csv', index_col=0, parse_dates=True)
    df_lulc_rel = pd.read_csv('significance_results/nonparametric/' + 'Rel_SOD_%' + '/' + str(win_size) + '_year_MA/'
                              + 'less' + '_multi_' + 'bylulc' + '_win' +
                              str(win_size) + '.csv', index_col=0, parse_dates=True)

    # load detection rates for flood volume sorted by gcm/rcp/lulc
    df_gcm_flood = pd.read_csv(
        'significance_results/nonparametric/' + 'Upstream_Flood_Volume_taf' + '/' + str(win_size) + '_year_MA/'
        + 'greater' + '_multi_' + 'bygcm' + '_win' +
        str(win_size) + '.csv', index_col=0, parse_dates=True)
    df_rcp_flood = pd.read_csv(
        'significance_results/nonparametric/' + 'Upstream_Flood_Volume_taf' + '/' + str(win_size) + '_year_MA/'
        + 'greater' + '_multi_' + 'byrcp' + '_win' +
        str(win_size) + '.csv', index_col=0, parse_dates=True)
    df_lulc_flood = pd.read_csv(
        'significance_results/nonparametric/' + 'Upstream_Flood_Volume_taf' + '/' + str(win_size) + '_year_MA/'
        + 'greater' + '_multi_' + 'bylulc' + '_win' +
        str(win_size) + '.csv', index_col=0, parse_dates=True)
    # create subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[10, 6])
    fig.suptitle('Distribution of detection rates in 2098 by GCM/RCP/LULC')

    df_gcm_rel.loc['2098-10-01', :].plot.hist(bins=10, ax=axes[0, 0])
    axes[0, 0].set_title('Water supply reliability by GCM')
    axes[0, 0].set_xlabel('detection rate')

    df_lulc_rel.loc['2098-10-01', :].plot.hist(bins=10, ax=axes[0, 1])
    axes[0, 1].set_title('Water supply reliability by LULC')
    axes[0, 1].set_xlabel('detection rate')

    # df_rcp_rel.loc['2098-10-01', :].plot.hist(bins=10, ax=axes[0, 2])
    axes[0, 2].bar(['rcp26', 'rcp45', 'rcp60', 'rcp85'], df_rcp_rel.loc['2098-10-01', :], width=0.4)
    axes[0, 2].set_title('Water supply reliability by RCP')
    axes[0, 2].set_ylabel('detection rate')
    axes[0, 2].set_ylim(top=1)

    df_gcm_flood.loc['2098-10-01', :].plot.hist(bins=10, ax=axes[1, 0])
    axes[1, 0].set_title('Upstream flood volume by GCM')
    axes[1, 0].set_xlabel('detection rate')

    df_lulc_flood.loc['2098-10-01', :].plot.hist(bins=10, ax=axes[1, 1])
    axes[1, 1].set_title('Upstream flood volume by LULC')
    axes[1, 1].set_xlabel('detection rate')
    # axes[1, 1].set_xlim(left=0, right=1)

    # df_rcp_flood.loc['2098-10-01', :].plot.hist(bins=10, ax=axes[1, 2])
    axes[1, 2].bar(['rcp26', 'rcp45', 'rcp60', 'rcp85'], df_rcp_flood.loc['2098-10-01', :], width=0.4)
    axes[1, 2].set_title('Upstream flood volume by RCP')
    axes[1, 2].set_ylabel('detection rate')
    axes[1, 2].set_ylim(top=1)

    plt.tight_layout()
    # plt.show()
    plt.savefig('significance_results/article_figures/detection_rate_sorted.png')
    plt.clf()

    return


# plot p-vals of 50 randomly selected scenarios
def plot_pvals(win_size=30):
    # import pvals, delete values before year 2000 (historical)
    rel_p_vals = pd.read_csv('significance_results/nonparametric/' + 'Rel_SOD_%' + '/' + str(win_size) +
                             '_year_MA/' + 'less' + '_pvals_win' + str(win_size) + '.csv', index_col=0,
                             parse_dates=True).drop(['count', 'rel_count'], axis=1)
    flood_p_vals = pd.read_csv('significance_results/nonparametric/' + 'Upstream_Flood_Volume_taf' + '/' + str(win_size)
                               + '_year_MA/' + 'greater' + '_pvals_win' + str(win_size) + '.csv', index_col=0,
                               parse_dates=True).drop(['count', 'rel_count'], axis=1)
    rel_p_vals = rel_p_vals['2000-10-1':'2098-10-1'].sample(n=50, axis=1, random_state=0)
    flood_p_vals = flood_p_vals['2000-10-1':'2098-10-1'].sample(n=50, axis=1, random_state=0)

    # plot pvals
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 5], sharey=True)
    rel_p_vals.plot(ax=axes[0], c='lightgray', legend=False)
    axes[0].set_ylabel('p-value')
    axes[0].set_title('Water supply reliability')
    flood_p_vals.plot(ax=axes[1], c='lightgray', legend=False)
    axes[1].set_title('Upstream flood volume')
    # plot p=0.05
    axes[0].axhline(y=0.05, color='r')
    axes[1].axhline(y=0.05, color='r')
    # create custom legend
    legend_elements = [Line2D([0], [0], color='lightgray', label='50 randomly selected scenarios'),
                       Line2D([0], [0], color='red', label='p=0.05'),
                       ]
    axes[0].legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('significance_results/article_figures/p_vals_win' + str(win_size) + '.png')
    plt.clf()

    return


# plot p-vals of scenarios that showed a detection and did not (25 randomly selected from each)
def pvals_filter(win_size=30):
    # import pvals, delete values before year 2000 (historical)
    rel_p_vals = pd.read_csv('significance_results/nonparametric/' + 'Rel_SOD_%' + '/' + str(win_size) +
                             '_year_MA/' + 'less' + '_pvals_win' + str(win_size) + '.csv', index_col=0,
                             parse_dates=True).drop(['count', 'rel_count'], axis=1)
    flood_p_vals = pd.read_csv('significance_results/nonparametric/' + 'Upstream_Flood_Volume_taf' + '/' + str(win_size)
                               + '_year_MA/' + 'greater' + '_pvals_win' + str(win_size) + '.csv', index_col=0,
                               parse_dates=True).drop(['count', 'rel_count'], axis=1)
    rel_p_vals = rel_p_vals['2000-10-1':'2098-10-1']
    flood_p_vals = flood_p_vals['2000-10-1':'2098-10-1']

    # randomly select 25 scenarios with detection
    rel_no_detect_names = pd.read_csv('significance_results/nonparametric/Rel_SOD_%/additional_materials/no_detect_'
                                      'scenarios_less_single_win30.csv',
                                      index_col=0).sample(n=25, axis=0, random_state=0)['Model'].tolist()
    flood_no_detect_names = pd.read_csv('significance_results/nonparametric/Upstream_Flood_Volume_taf/'
                                        'additional_materials/no_detect_scenarios_greater_single_win30.csv',
                                        index_col=0).sample(n=25, axis=0, random_state=0)['Model'].tolist()
    rel_p_vals_no_detect = rel_p_vals[rel_no_detect_names]
    flood_p_vals_no_detect = flood_p_vals[flood_no_detect_names]

    # randomly select 25 scemarios without detection
    rel_p_vals_detect = rel_p_vals.drop(rel_no_detect_names, axis=1).sample(n=25, axis=1, random_state=0)
    flood_p_vals_detect = flood_p_vals.drop(flood_no_detect_names, axis=1).sample(n=25, axis=1, random_state=0)

    # plot pvals
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 5], sharey=True)
    rel_p_vals_detect.plot(ax=axes[0], c='wheat', legend=False)
    rel_p_vals_no_detect.plot(ax=axes[0], c='lightgray', legend=False)
    axes[0].set_ylabel('p-value')
    axes[0].set_title('Water supply reliability')

    flood_p_vals_detect.plot(ax=axes[1], c='wheat', legend=False)
    flood_p_vals_no_detect.plot(ax=axes[1], c='lightgray', legend=False)
    axes[1].set_title('Upstream flood volume')

    # plot p=0.05
    axes[0].axhline(y=0.05, color='r')
    axes[1].axhline(y=0.05, color='r')

    # create custom legend
    legend_elements = [Line2D([0], [0], color='wheat', label='scenarios with detection'),
                       Line2D([0], [0], color='lightgray', label='scenarios without detection'),
                       Line2D([0], [0], color='red', label='p=0.05'),
                       ]
    axes[1].legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('significance_results/article_figures/p_vals_win' + str(win_size) + '_filtered.png')
    plt.clf()

    return


# Single scenario: Plot first detection year against objective value (moving average) at END OF PROJECTION (2098)
# INCLUDES DISTRIBUTION OF NO DETECT SCENAIORS' SEVERITY (BOXPLOT)
def detect_vs_end_obj(objective, win_size=30):
    # set alternative and figure title based on objective
    if objective == 'Rel_SOD_%':
        alt = 'less'
        name = 'water supply reliability'
    else:
        alt = 'greater'
        name = 'upstream flood volume'

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
    axs[0].set_ylabel(name)
    boxplot['fliers'][0].set(markersize=4, mec='black')
    boxplot['boxes'][0].set(linewidth=1, color='black')
    boxplot['medians'][0].set(linewidth=1, color='black')
    for cap in boxplot['caps']:
        cap.set(linewidth=1, color='black')
    for whisker in boxplot['whiskers']:
        whisker.set(linewidth=1, color='black')

    plt.savefig(
        'significance_results/article_figures/' + objective + '_detect_2098obj_' + alt + '_single_'
        + 'win' + str(win_size) + '.svg')
    plt.clf()

    return output_df


def main():
    # # create ensemble subplots
    # plot_ensemble()

    # # plot distribution of first detection
    # plot_single_total()

    # # plot statistics of first detection sorted by gcm/rcp/lulc
    # plot_single_sorted('Rel_SOD_%')
    # plot_single_sorted('Upstream_Flood_Volume_taf')

    # # plot detection rates
    # plot_multi_total()

    # # plot detection rate sorted by gcm/rcp/lulc in 2098
    # plot_multi_sorted()

    # # plot p-vals
    # plot_pvals(win_size=30)

    # plot filtered p-vals
    pvals_filter()

    # # plot detection vs. end of simulation objective severity
    # detect_vs_end_obj('Rel_SOD_%')
    # detect_vs_end_obj('Upstream_Flood_Volume_taf')

    return


if __name__ == "__main__":
    main()
