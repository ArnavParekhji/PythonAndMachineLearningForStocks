import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import quandl
from quandl.errors.quandl_error import NotFoundError
import pickle
import requests

resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', class_ = 'wikitable sortable')
tickers = []
names = []
for row in table.find_all('tr')[1:]:
    ticker = row.find_all('td')[1].text
    name = row.find_all('td')[0].text
    tickers.append(ticker)
    names.append(name)

df = pd.DataFrame()
df['Names'] = names
df['Tickers'] = tickers

df.to_csv('sp500tickersandnames.csv')
