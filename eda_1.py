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
#
# statsmodels.graphics.tsaplots.plot_acf(x=nasdaq_vals)
# plt.show()
#
# statsmodels.graphics.tsaplots.plot_pacf(x=nasdaq_vals,lags=5)
# plt.show()



# result = seasonal_decompose(series, model='multiplicative', freq=1)
# result.plot()
# plt.show()

print statsmodels.tsa.stattools.adfuller(nasdaq_vals,regression='ctt')

f2, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
# f2.tight_layout()
series = df[nasdaq_attr]
autocorrelation_plot(series, ax=ax1)
ax1.set_title('Values of NASDAQ - Lag plot')
plt.show()