import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.stattools
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot



df = pd.read_csv('data/nasdaq100_padding.csv',header=0,index_col=None)
print df.columns
nasdaq_attr = 'NDX'
nasdaq_vals = list(df[nasdaq_attr])
time = range(len(nasdaq_vals))

# plt.figure()
# plt.title('Value of NDX (nasdaq 100) over time')
# plt.plot(time, nasdaq_vals, 'r-')
# plt.show()

statsmodels.graphics.tsaplots.plot_acf(x=nasdaq_vals)
plt.show()

statsmodels.graphics.tsaplots.plot_pacf(x=nasdaq_vals,lags=5)
plt.show()


# series = df[nasdaq_attr]
# result = seasonal_decompose(series, model='multiplicative', freq=1)
# result.plot()
# plt.show()

# print statsmodels.tsa.stattools.adfuller(nasdaq_vals,regression='ctt')
#
# f2, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
# # f2.tight_layout()
# series = df[nasdaq_attr]
# autocorrelation_plot(series, ax=ax1)
# ax1.set_title('Values of NASDAQ - Lag plot')
# plt.show()
#


# from random import randrange
# from pandas import Series
# from matplotlib import pyplot
# from statsmodels.tsa.seasonal import seasonal_decompose
# series = nasdaq_vals
# result = seasonal_decompose(series, model='multiplicative', freq=1)
# import numpy as np
# import math
# import matplotlib.patches
# import matplotlib.patches as mpatches
#
# # result.plot()
# # pyplot.show()
# x = range(len(nasdaq_vals))
# y= nasdaq_vals
# p_order = 3
# z = np.polyfit(x,y,p_order)
# print z
# x1 = []
# for t in x:
#     res = z[0]*t*t*t + z[1]*t*t + z[2]*t + z[3]
#     x1.append(res)
# plt.title('Trend Estimation in data. ')
# plt.plot(x,x1,'r-')
# plt.xlabel('Time points',fontsize = 20)
# plt.plot(x,y,'b-')
# red = mpatches.Patch(color='red', label='Fitted Trend Line')
# blue = mpatches.Patch(color='blue', label='Real Data Values')
# patches = [ red, blue]
# plt.legend(handles=patches)
# plt.show()
