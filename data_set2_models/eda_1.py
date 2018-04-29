import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.stattools
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from random import randrange
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import math
import matplotlib.patches
import matplotlib.patches as mpatches

df = pd.read_csv('dataset2/weather.csv',header=0,index_col=None)
print df.columns
end_attr = 'temp'
temp_vals = list(df[end_attr])
time = range(len(temp_vals))

plt.figure()
plt.title('Value of Temperature')
plt.plot(time, temp_vals, 'r-')
plt.show()

# statsmodels.graphics.tsaplots.plot_acf(x=temp_vals)
# plt.show()
#
# statsmodels.graphics.tsaplots.plot_pacf(x=temp_vals,lags=5)
# plt.show()


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


#
# series = temp_vals
# result = seasonal_decompose(series, model='additive', freq=1440)
# result.plot()
# pyplot.show()


#
# # result.plot()
# # pyplot.show()
x = range(len(temp_vals))
y= temp_vals
p_order = 3
z = np.polyfit(x,y,p_order)
print z
x1 = []
for t in x:
    res = z[0]*t*t*t + z[1]*t*t + z[2]*t + z[3]
    x1.append(res)
plt.title('Trend Estimation in data. ')
plt.plot(x,x1,'r-')
plt.xlabel('Time points',fontsize = 20)
plt.plot(x,y,'b-')
red = mpatches.Patch(color='red', label='Fitted Trend Line')
blue = mpatches.Patch(color='blue', label='Real Data Values')
patches = [ red, blue]
plt.legend(handles=patches)
plt.show()
