FUNCTION: Determine if modeled reservoir system performance is statistically significant compared to historical data

For SSJRB model outputs, use MWU test against historical to find years of significnance. A single scenario analysis would seek to find the 
first statistically significant detection year for each scenario, while a multiple sceanrio counts detection across
multiple models for each year. Results ares sorted by GCM/RCP choice (CMIP5 scenarios) and land use (LULC scenario). 

significance_detection_v4.py defined functions for multiple/single scenario analysis
plot_results.py *RUN FIRST TO GET NECCESSARY FILES* defines and executes functions to plot and export results (csv) based on significance detection functions
detection_classifier.py uses logistic regression to predict detections within lead time L years after year t*. Performance graded with f1 scores, and compared to
a dummy random classifier. Choises of L and t* flexible.
plot_csv.py exports additional tables and plots of csv files exported by plot_results.py
sobol-sensitivity.py runs sobol sensitivity analysis on output (single scenario) csv files, also exports information on no exist/no detection scenarios
plot_ensemble.py plots ensmeble of all model scenarios

FOLDERS:
archive/ duplicate or uneeded files
data/ stores SSJRB model outputs in zip format
empty/ stores empty csv with relevant datatimes to help build df's
significance_results/ stores results. *See README within
cimp5_scenario_names.csv stores cimp5 names (gcm/rcp combinations)
lulc_scenario_names.csv stores land use scenarios (lulc)
obj_historical.csv objective values based on observed historical data

GENERAL FILE NAMING CONVENTION FOR RESULTS:
greater/lesser: type of one-tailed hypothesis test used
multi: multiple scenario analysis (e.g. count significant detection across multiple models for each year)
single: single scenario analysis (e.g. for each scenario, what is the first year of detection?)
win(num): size of rolling window used in analysis
byrcp/bygcm/bylulc: indicates by which scenario types results are sorted (total/no label = all scenarios)

NOTES:
Copy of SSJRB data: https://drive.google.com/file/d/1ZGVAbwcIbdQS1wmuHeZWKkiAEgK5GYPu/view?usp=sharing
Total detection "counts" are relative (rel_counts), presented as a fraction of total relevant scenarios