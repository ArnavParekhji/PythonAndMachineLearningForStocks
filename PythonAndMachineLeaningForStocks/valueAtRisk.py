import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style

style.use('ggplot')
    
#Create Multiple Portfolios
tickers = ["AAPL", "T", "JPM", "WFC", "V"]
 
#Weights in each stock
weights = np.array([.2, .2, .2, .2, .2])
 
#initial investment
initial_investment = 100000
 
#Override API
#yf.pdr_override()
data = pd.DataFrame()
 
# download Data
#data = pdr.get_data_yahoo(tickers, start="2017-01-01", end=dt.date.today())['Close']
for ticker in tickers:
    df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
    data[ticker] = df['Adj. Close']
 
#Calculate periodic returns
returns = data.pct_change()
 
#Generate Var-Cov matrix
cov_matrix = returns.cov()
 
#Calculate mean returns for each stock
avg_rets = returns.mean()
 
#Calculate Portfolio Mean
port_mean = avg_rets.dot(weights)
 
#Calculate Portfolio STDEV
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
 
#Mean Investment
mean_investment = (1+port_mean) * initial_investment
             
#Standard Deviation Investment
stdev_investment = initial_investment * port_stdev
 
#Cutoff Point 
conf_level0 = 0.001
conf_level1 = 0.01
conf_level2 = 0.05
 
from scipy.stats import norm
 
cutoff0 = norm.ppf(conf_level0, mean_investment, stdev_investment)
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)
cutoff2 = norm.ppf(conf_level2, mean_investment, stdev_investment)
 
#Calculate PDF (If Desired)
#print(norm.cdf(cutoff0, mean_investment, stdev_investment))
 
#Calculate 1 Day VaR at different confidence intervals 
var_1d0 = initial_investment - cutoff0
var_1d1 = initial_investment - cutoff1
var_1d2 = initial_investment - cutoff2
 
#-------------------------Optional--------------------------#
#Calculate n Day VaR in loop
num_days = 100
for x in range(1, num_days):    
    print(str(x) + " day VaR at 99.99% confidence level: " + str(np.round(var_1d0 * np.sqrt(x),2)))
    
#Plot Results On Bar Chart
 
data = {'99.99%': np.round(var_1d0,2), '99%': np.round(var_1d1, 2), '95%': np.round(var_1d2, 2)}
df = pd.DataFrame(data,index=[0]) 
 
df.plot(kind = 'bar')
plt.suptitle('Portfolio VaR', fontsize=20)
plt.ylabel('VaR ($)', fontsize=16)
plt.show()   
