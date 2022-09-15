IN THIS FOLDER:
no_exist_scenarios.csv: 
list of gcm/rcp/lulc scenarios that do not exist in the dataset

no_exist_cmip5.csv: 
notice that all gcm/rcp combinations have all lulc's. list of gcm/rcp combinations that do not exist
in the dataset

sobol_sensitivity.xlsx:
summarizes sobol sensitivity analysis for all objectives using 30 yr MA results


FOLDERS:
nonparametric/
main result outputs. please refer to GENERAL NAMING CONVENTIONS

nonparametric/*objective*/*MA window size*/additional materals/
additional tables/plots based on 30 year MA results and sobol sensitivity

parametric/
retired results from t-test

GENERAL NAMING CONVENTIONS:
greater/lesser: type of one-tailed hypothesis test used
multi: multiple scenario analysis (e.g. count significant detection across multiple models for each year)
single: single scenario analysis (e.g. for each scenario, what is the first year of detection?)
win(num): size of rolling window used in analysis
byrcp/bygcm/bylulc: indicates by which scenario types results are sorted (total/no label = all scenarios)