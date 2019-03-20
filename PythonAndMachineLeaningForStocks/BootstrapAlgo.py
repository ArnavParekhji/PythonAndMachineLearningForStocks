import pandas as pd
import quandl
import math
import datetime as dt
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import random
import alpha_vantage
import requests
from alpha_vantage.timeseries import TimeSeries
import json, csv, os
import time
import pandas_datareader.data as web

quandl.ApiConfig.api_key = "McjQvuKdficVT74V1rzw"
prices = pd.DataFrame()

class bootstrap:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def get_data(self, symbols): 
        start = self.start
        end = self.end

        tickers = symbols
        for ticker in tickers:
        #prices = web.DataReader(symbols, 'iex',start, end)['Close']
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            prices[ticker] = df['Adj. Close']
        
        #Get % Returns
        returns = prices.pct_change()
        
        #Drop NaN Values
        returns = returns[1:]
        
        #Get Last Closing Prices
        last_prices = prices.iloc[-1, :]
        last_price = prices.iloc[-1, :].sum()
 
        self.returns = returns
        self.last_prices = last_prices
        self.last_price = last_price


    def simulation(self, num_simulations):
            
            last_prices = self.last_prices
            returns = self.returns
        
            main_df = pd.DataFrame()
                
            for x in range(num_simulations):
            
                #Set First Values In Simulation
                pos = 0 
                
                first_values = []
                for price in last_prices:
                    ret = returns.iloc[0, pos]
                    value = price * math.exp(ret)
                    pos += 1
                    first_values.append(value)
                
                if x == 1:
                    #Drop First Row Again Since Used With First Values 
                    returns = returns[1:]
            
                #Empty Dataframe for Simulation
                simulation_df = pd.DataFrame()
            
                
                count = 0
            
                for column in returns:
                    
                    temp = []
                    
                    count1 = 0
                
                    col = (returns[column])
                
                    #Create Temporary Series to Append to Data Frame
                
                    #Append First Value
                    fv = first_values[count]
                    temp.append(fv)
                
                    count += 1
                    
                    #print temp
                    
                    for val in col:
                        if count1 == 249:
                            break
                            
                        f_price = temp[count1] * math.exp(val)        
                        
                        temp.append(f_price)
                        count1 += 1
                                
                    simulation_df[count] = temp
                
                simulation_df['Sum'] = simulation_df.sum(axis=1)
                #result = simulation_df['Sum'].values[-1]
                trial = simulation_df['Sum']
                #results.append(result)
                
                #Randomly Re-Order and Re-Index for Next Trial
                returns = returns.sample(frac=1).reset_index(drop=True)
                main_df[x] = trial
                       
            self.main_df = main_df


    def analyze(self):            
        main_df = self.main_df
        last_price = self.last_price 
     
        fig = plt.figure()
        title = "Bootstrap Historical Simulation"
        plt.plot(main_df)
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Price ($USD)')
        plt.grid(True,color='grey')
        plt.axhline(y=last_price, color='r', linestyle='-')
        plt.show()


if __name__== "__main__":
    start = dt.datetime(2017, 1, 3)
    end = dt.datetime(2017, 10, 4)
    sim = bootstrap(start, end)
    
    symbols = ['AAPL', 'INTC', 'JPM', 'GM', 'F']
    #for symbol in symbols:
    sim.get_data(symbols)
    sim.simulation(100)
    sim.analyze()
