import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import quandl

style.use('ggplot')
quandl.ApiConfig.api_key = "McjQvuKdficVT74V1rzw"

start = dt.datetime(2017,1,3)
end = dt.datetime(2018,1,3)

df_tick = pd.read_csv('sp500tickersandnames.csv')
company_name = input("Enter: ")
i = 0
for name in df_tick['Names']:
    if name == company_name:
        n = df_tick['Tickers'][i]
        break
    i += 1

print(n)

company = 'WIKI/' + n
df = quandl.get(company, start_date = start, end_date = end)
prices = df['Adj. Close']
returns = prices.pct_change()
last_price = prices[-1]

num_simulations = 1000
num_days = 252

greater = 0
lesser = 0

simulation_df = pd.DataFrame()

for x in range(num_simulations):
    count = 0
    daily_volatility =  returns.std()

    price_series = []

    price = last_price * (1 +  np.random.normal(0, daily_volatility))
    price_series.append(price)

    for y in range(num_days):
        if count == 251:
            break
        price = price_series[count] * (1 +  np.random.normal(0, daily_volatility))
        if price > last_price:
            greater += price-last_price
        elif price < last_price:
            lesser += last_price-price
        price_series.append(price)
        count += 1

    simulation_df[x] = price_series


print(greater, lesser)
pct_greater = (greater/(greater+lesser)) * 100
pct_lesser = (lesser/(greater+lesser)) * 100
print('%Loss:', pct_lesser, '%Profit:', pct_greater)
fig = plt.figure()
plt.plot(simulation_df)
fig.suptitle('Monte Carlo Simulation: ' + company_name)
plt.axhline(y = last_price, color = 'r', linestyle = '-')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()
