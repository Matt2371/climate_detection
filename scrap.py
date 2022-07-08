import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10,4])
obs = pd.read_csv('obj_historical.csv', index_col=0, parse_dates=True)
obs.loc['2020-10-1', :] = obs.loc['2000-10-1':'2020-10-1'].mean(axis=0)
obs['datetime'] = obs.index

print(obs.loc['2020-10-1', 'datetime'])

axes[0].plot(obs.loc['2020-10-1', 'datetime'], obs.loc['2020-10-1', 'Rel_SOD_%'], c='w', marker='o', markerfacecolor='red', markersize=10)
axes[1].plot(obs.loc['2020-10-1', 'datetime'], obs.loc['2020-10-1', 'Upstream_Flood_Volume_taf'],
                    c='w', marker='o', markerfacecolor='red', markersize=10)
plt.tight_layout()
plt.show()