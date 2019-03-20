import pandas as pd
import quandl
import math
import datetime as dt
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pickle
import pandas_datareader.data as web

style.use('ggplot')

##quandl.ApiConfig.api_key = "McjQvuKdficVT74V1rzw"
##
##start = dt.datetime(2000,1,1)
##end = dt.datetime(2016,12,31)
##
##df = quandl.get('WIKI/GOOGL', start_date = start, end_date = end)
##df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj. Close']]
##df.to_csv('google.csv')

df = pd.read_csv('google.csv', parse_dates = True, index_col = 0)
#df['100ma'] = df['Adj. Close'].rolling(window = 100, min_periods = 0).mean()

df_ohlc = df['Adj. Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace = True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()
