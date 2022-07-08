import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

### for a given objective, plot entire ensemble


# store names of gcm/rcp/lulc combinations
gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
            'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
            'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
            'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
            'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']

rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']

lulc_names = pd.read_csv('lulc_scenario_names.csv').name.to_list()


def ensemble(objective, win_size=30):
    # import empty df with dates
    # df = pd.read_csv('empty/datetime.csv', index_col=0,
    #                  parse_dates=True)
    df = pd.DataFrame()

    for gcm in tqdm(gcm_list, desc= 'Reading data'):
        for rcp in rcp_list:
            for lulc in lulc_names:
                try:
                    # read all scenarios, plot rolling average
                    scenario = gcm + '_' + rcp + '_r1i1p1'
                    data_df = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0,
                                          parse_dates=True)
                    df[gcm + '_' + rcp + '_' + lulc] = data_df[objective].rolling(win_size).mean()
                except FileNotFoundError:
                    pass
    df['mean'] = df.mean(axis=1)

    return df


def main():
    # list of objectives
    obj_list = ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs']

    # save plots for all objectives
    for objective in tqdm(obj_list, desc='Plotting ensemble'):
        df = ensemble(objective)
        df['1981-10-1':'2098-10-1'].plot(legend=False, c='lightgray')
        df.loc['1981-10-1':'2098-10-1', 'mean'].plot(c='steelblue')
        plt.ylabel('objective')
        plt.title(objective + ', 30 yr MA')

        plt.savefig('significance_results/nonparametric/' + objective + '/ensemble_plot.png')
        plt.clf()
    return


if __name__ == "__main__":
    main()
