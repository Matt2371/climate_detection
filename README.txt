Detecting climate change effects in water resources is critical for finding signals in the data upon which adaptation decisions can be triggered. 
To study this, we explore if changes in projected water resources are statistically significant using a nonparametric hypothesis test, the Mann Whitney U-test (MWU). 
We are also interested in our ability to predict when statistically significant changes occur. 
To do so, we train a logistic regression classifier to predict if a significant change occurs within a specified lead time.

For water resources projections into the next 100 years from the SSJRB model, we conduct the MWU test on the historical data vs. 
30-year rolling windows on the projected data to find if and when we detect significant changes. 
Here, we define the single scenario analysis as finding the first statistically significant detection year for each scenario, 
while a multiple sceanrio would track detection rates across multiple models for each year. We sort results by choice of climate model (GCM), or 
emissions scenario (RCP) choice (CMIP5 scenarios) and land use (LULC scenario) to see which of these factors are detections most sensitive to. 
This was also explored more formally with a numeric global sensitivity analysis.

Code Files:
significance_detection_v4.py: defined functions for multiple/single scenario analysis

plot_results.py: *RUN FIRST TO GET NECCESSARY FILES* defines and executes functions to plot and export results (csv) based on significance detection functions

detection_classifier.py: uses logistic regression to predict detections within lead time L years after year t*. 
Performance graded with true positive and true negative scores. Results were validated using a 75/25 train/test split.
Choices of L and t* flexible.

plot_csv.py: exports additional tables and plots of csv files exported by plot_results.py

sobol-sensitivity.py: runs sobol sensitivity analysis on output (single scenario) csv files, also exports information on no exist/no detection scenarios

plot_ensemble.py: plots ensmeble of all model scenarios

article_figs.py: creates figures for journal manuscript *logistic regression figures plotted seperately

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
_expanding: indicates expanding window analysis (instead of rolling). FLOOD OBJECTIVE ONLY
_pw: indicates the use of the pre-whitening method to remove lag-1 autocorrelation'

NOTES:
Copy of SSJRB data: https://drive.google.com/file/d/1ZGVAbwcIbdQS1wmuHeZWKkiAEgK5GYPu/view?usp=sharing
Total detection "counts" are relative (rel_counts), presented as a fraction of total relevant scenarios
