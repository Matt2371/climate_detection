import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


# take rolling averages, run statistical significance tests against historical
# paramters: gcm, rcp, objective (reliability/flood)
def rolling_significance(gcm, rcp, objective):
    # load dataframe and isolate objective
    scenario = gcm + '_' + rcp + '_r1i1p1'
    df = pd.read_csv('data/%s-results.csv' % scenario, index_col=0, parse_dates=True)
    df = df[[objective]]

    # get rolling samples (future)
    df['rolling'] = [window.to_list() for window in df.reliability.rolling(window=3)]
    df.loc['1951-10-01':'1952-10-01', 'rolling'] = float("NaN")
    print(df)
    #ask pHD students about df.loc['1951-10-01':'1952-10-01', 'rolling'] vs df.loc['1951-10-01':'1952-10-01'].rolling


# Fix GCM/RCP
gcm = 'noresm1-m'
rcp = 'rcp26'
objective = 'reliability'

rolling_significance(gcm, rcp, objective)
