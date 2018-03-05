import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots

df = pd.read_csv('data/nasdaq100_padding.csv',header=0,index_col=None)
print df.columns
nasdaq_attr = 'NDX'
nasdaq_vals = list(df[nasdaq_attr])
time = range(len(nasdaq_vals))

plt.figure()
plt.title('Value of NDX (nasdaq 100) over time')
plt.plot(time, nasdaq_vals, 'r-')
plt.show()

statsmodels.graphics.tsaplots.plot_acf(x=nasdaq_vals)
plt.show()

statsmodels.graphics.tsaplots.plot_pacf(x=nasdaq_vals,lags=5)
plt.show()

