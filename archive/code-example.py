import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

scenario = 'access1-0_rcp85_r1i1p1'
df = pd.read_csv('data/%s-results.csv' % scenario, index_col=0, parse_dates=True)
print(df) # dataframe object, indexed by date

df.reliability.plot()
# rolling function, and chaining function calls
df.reliability.rolling(30, center=True).mean().plot(color='k')
plt.legend(['Reliability', '30-year rolling average'])
plt.show()

