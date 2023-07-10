from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd


### Find how gcm/rcp/lulc controls variability in results: use sobol sensitivity analysis
### Also returns information about dataset (what scenarios don't exist, how many no detections for single scenario)
## Need csv results from plot_results.py
## Stored in /additional_materials/ folders

## sobol code and examples: https://github.com/salib/salib
## sobol full docs: https://salib.readthedocs.io/en/latest/getting-started.html

## Shared parameters:
## objective: ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs'], str
## win_size: moving average window size applied, int
## alt = ['two-sided', 'less', 'greater']

def sobol_detection(objective, alt, win_size, pre_whitening):
    """
    Sobol sensitivity analysis for first detection years, parameters consistent with plot_csv.py and above
    """
    problem = {
        'num_vars': 3,
        'names': ['rcp', 'gcm', 'lulc'],
        'bounds': [[0, 4], [0, 31], [0, 36]]}

    # Generate samples
    X = saltelli.sample(problem, 1000)
    X = X.astype(int)  # scenario numbers are integers

    # Store scenario names
    gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
                'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
                'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
                'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
                'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']
    rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    lulc_list = pd.read_csv('lulc_scenario_names.csv').name.to_list()

    # initiate list of objective values (eventually converted to numpy vector)
    Y_list = []


    # import first detection years
    if pre_whitening:
        filename = f'significance_results/nonparametric/{objective}/{str(win_size)}_year_MA/{alt}_single_total_win{str(win_size)}_pw.csv'
    else:
        filename = f'significance_results/nonparametric/{objective}/{str(win_size)}_year_MA/{alt}_single_total_win{str(win_size)}.csv'

    data = pd.read_csv(filename, index_col='Model')

    # take those three values and find the detection year of interest, save in vector Y
    for scenario in X:
        rcp_number = scenario[0]
        gcm_number = scenario[1]
        lulc_number = scenario[2]

        # convert scenario numbers to actual names
        gcm = gcm_list[gcm_number]
        rcp = rcp_list[rcp_number]
        lulc = lulc_list[lulc_number]

        # read detection year, add to Y
        # SCENARIO DOES NOT EXIST: ADD MEAN VALUE (2049)
        # NO DETECTION (DETECTION YR >2100): ADD END OF PROJ (2100)
        try:
            scenario = gcm + '_' + rcp + '_' + lulc
            detect = int(data.loc[scenario]['Year'])
            Y_list.append(detect)
        except KeyError:
            Y_list.append(2049)
        except ValueError:
            Y_list.append(2100)

    # convert Y to vector
    Y = np.array(Y_list)

    # Perform sensitivity analysis
    # Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
    # (first and total-order indices with bootstrap confidence intervals)
    Si = sobol.analyze(problem, Y, print_to_console=True)

    # convert results to df and save
    total_Si, first_Si, second_Si = Si.to_df()
    if pre_whitening:
        save_dir = f'significance_results/nonparametric/{objective}/additional_materials/sobol_pw/'
    else:
        save_dir = f'significance_results/nonparametric/{objective}/additional_materials/sobol/'

    total_Si.to_csv(save_dir + 'sobol_ST_' + alt + '_single_win' + str(win_size) + '.csv')
    first_Si.to_csv(save_dir + 'sobol_S1_' + alt + '_single_win' + str(win_size) + '.csv')
    second_Si.to_csv(save_dir + 'sobol_S2_' + alt + '_single_win' + str(win_size) + '.csv')

    return Si


## Export csv of scenario names that don't exist
def scenario_nonexist():
    # Store scenario names
    gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
                'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
                'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
                'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
                'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']
    rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    lulc_list = pd.read_csv('lulc_scenario_names.csv').name.to_list()
    # initiate list of scenarios that don't exist, also for gcm/rcp only (cmip5 scenario)
    no_exist_scenarios = []
    no_exist_cmip5 = []
    # import ANY p-value table from results
    df = pd.read_csv('significance_results/nonparametric/Rel_NOD_%/30_year_MA/less_pvals_win30.csv', index_col=0,
                     parse_dates=True)
    # get scenario names (that do exist)
    scenarios = list(df)
    # check every gcm/rcp/lulc combination
    for gcm in gcm_list:
        for rcp in rcp_list:
            for lulc in lulc_list:
                scenario = gcm + '_' + rcp + '_' + lulc
                cmip5 = gcm + '_' + rcp
                if scenario in scenarios:
                    pass
                else:
                    no_exist_scenarios.append(scenario)
                    if cmip5 not in no_exist_cmip5:
                        no_exist_cmip5.append(cmip5)
    # export results
    data1 = {'Model': no_exist_scenarios}
    data2 = {'Model': no_exist_cmip5}
    output1 = pd.DataFrame(data1)
    output2 = pd.DataFrame(data2)
    output1.to_csv('significance_results/no_exist_scenarios.csv')
    output2.to_csv('significance_results/no_exist_cmip5.csv')

    return


## Get no detection scenarios
# NEED TO RUN scenario_nonexist() first
def scenario_no_detect(objective, alt, win_size, pre_whitening):
    # initiate list of scenarios that didn't detect, also for gcm/rcp only (cmip5 scenario)
    no_detect_scenarios = []

    # import total single detections from results
    if pre_whitening:
        filename = f'significance_results/nonparametric/{objective}/{str(win_size)}_year_MA/{alt}_single_total_win{str(win_size)}_pw.csv'
    else:
        filename = f'significance_results/nonparametric/{objective}/{str(win_size)}_year_MA/{alt}_single_total_win{str(win_size)}.csv'

    df = pd.read_csv(filename, index_col=0)

    # iterate over detection years, row stores index, 'model' (scenario name), 'Year' (detection year)
    for row in df.itertuples():
        if np.isnan(row[2]):
            no_detect_scenarios.append(row[1])
        else:
            continue

    # export data
    data = {'Model': no_detect_scenarios}
    output = pd.DataFrame(data)
    # Save results
    if pre_whitening:
        save_dir = f'significance_results/nonparametric/{objective}/additional_materials/no_detect_scenarios_{alt}_single_win{str(win_size)}_pw.csv'
    else:
        save_dir = f'significance_results/nonparametric/{objective}/additional_materials/no_detect_scenarios_{alt}_single_win{str(win_size)}.csv'

    output.to_csv(save_dir)

    return output


## FOR EXPANDING WINDOW/FLOODING ONLY
## Sobol sensitivity analysis for first detection years, parameters consistent with plot_csv.py and above
def sobol_detection_expanding():
    problem = {
        'num_vars': 3,
        'names': ['rcp', 'gcm', 'lulc'],
        'bounds': [[0, 4], [0, 31], [0, 36]]}

    # Generate samples
    X = saltelli.sample(problem, 1000)
    X = X.astype(int)  # scenario numbers are integers

    # Store scenario names
    gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
                'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
                'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
                'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
                'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']
    rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    lulc_list = pd.read_csv('lulc_scenario_names.csv').name.to_list()

    # initiate list of objective values (eventually converted to numpy vector)
    Y_list = []
    # import first detection years
    data = pd.read_csv('significance_results/nonparametric/Upstream_Flood_Volume_taf/expanding_window/'
                       'expanding_window_single_total.csv', index_col='Model')
    # take those three values and find the detection year of interest, save in vector Y

    for scenario in X:
        rcp_number = scenario[0]
        gcm_number = scenario[1]
        lulc_number = scenario[2]

        # convert scenario numbers to actual names
        gcm = gcm_list[gcm_number]
        rcp = rcp_list[rcp_number]
        lulc = lulc_list[lulc_number]

        # read detection year, add to Y
        # SCENARIO DOES NOT EXIST: ADD MEAN VALUE (2049)
        # NO DETECTION (DETECTION YR >2100): ADD END OF PROJ (2100)
        try:
            scenario = gcm + '_' + rcp + '_' + lulc
            detect = int(data.loc[scenario]['Year'])
            Y_list.append(detect)
        except KeyError:
            Y_list.append(2049)
        except ValueError:
            Y_list.append(2100)

    # convert Y to vector
    Y = np.array(Y_list)

    # Perform sensitivity analysis
    # Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
    # (first and total-order indices with bootstrap confidence intervals)
    Si = sobol.analyze(problem, Y, print_to_console=True)

    # convert results to df and save
    total_Si, first_Si, second_Si = Si.to_df()
    total_Si.to_csv('significance_results/nonparametric/Upstream_Flood_Volume_taf/expanding_window/'
                    'expanding_window_sobol_ST.csv')
    first_Si.to_csv('significance_results/nonparametric/Upstream_Flood_Volume_taf/expanding_window/'
                    'expanding_window_sobol_S1.csv')
    second_Si.to_csv('significance_results/nonparametric/Upstream_Flood_Volume_taf/expanding_window/'
                     'expanding_window_sobol_S2.csv')

    return Si


# Run script
def main():
    scenario_nonexist()

    # export results for each objective
    obj_list = ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf']
    win_size = 30

    for objective in obj_list:
        for pre_whitening in [True, False]:
            if objective in ['Rel_NOD_%', 'Rel_SOD_%']:
                alt = 'less'
            if objective == 'Upstream_Flood_Volume_taf':
                alt = 'greater'
            sobol_detection(objective, alt, win_size, pre_whitening)
            scenario_no_detect(objective, alt, win_size, pre_whitening)

    # # run sobol analysis for expanding window floods
    # sobol_detection_expanding()

    return

if __name__ == "__main__":
    main()


# sobol_detection('Upstream_Flood_Volume_taf', 'greater', 30)
# scenario_no_detect('Rel_NOD_%', 'less', 30)
#
# things to think about:
# 1. what is the output of interest (Y)? detection time? objective value in a certain year?
# for the detection time, only one SA would be needed per objective
# for the objective values, it could be a timeseries of % variance like Lehner et al. (2020)

# 2. what to do for scenario combinations that don't exist? fill with mean value?
