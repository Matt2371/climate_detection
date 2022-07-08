import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fix GCM
GCM = 'noresm1-m'


rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
scenarios_list = []

# Plot Reliability for different RCP's (rolling average)
for rcp in rcp_list:
    scenarios_list.append(GCM+'_'+rcp+'_r1i1p1')

for scenario in scenarios_list:
    df = pd.read_csv('data/%s-results.csv' % scenario, index_col=0, parse_dates=True)
    print(df)  # dataframe object, indexed by date

    df.reliability.rolling(30, center=True).mean().plot()
    # df.reliability.plot()
plt.title('Reliability')
plt.legend(rcp_list)
plt.show()

# Plot Flood for different RCP's (rolling average)
for scenario in scenarios_list:
    df = pd.read_csv('data/%s-results.csv' % scenario, index_col=0, parse_dates=True)
    print(df)  # dataframe object, indexed by date

    df.flood.rolling(30, center=True).mean().plot()
    # df.reliability.plot()
plt.title('Flooding')
plt.legend(rcp_list)
plt.show()

