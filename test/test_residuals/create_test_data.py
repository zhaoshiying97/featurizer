# -*- coding: utf-8 -*-

# Author: Huanqiu Wang
# Created on: Nov 6, 2020

'''
This module contains the code that creates data for test cases, including
    - data needed while running the unit tests
    - the expected input data
    - the expected output data to be compared with the dynamic outputs
    
Note: If you want to re-run this file to create a different set of data, please make sure 
      that the functions and functors involved are correct, so that the resulting datasets
      are not erroneous themselves

'''

import pandas as pd
import pickle
import torch
import numpy as np
import get_securities_fields_tensors as gs


########################### create mock x and y dataframes #####################

num_days = 20
datetime = pd.date_range(start= '2020-10-01', periods= num_days, freq= 'D')
datetime_stocks = datetime.append(datetime)
datetime_index = datetime

stocks = ['000001.XSHE', '000002.XSHE'] * num_days
stocks.sort()

index = ['399006.XSHE'] * num_days

close_stock1 = np.random.uniform(low= 10, high= 11, size= num_days)
close_stock2 = np.random.uniform(low= 20, high= 21, size= num_days)
close_stocks = np.hstack((close_stock1, close_stock2)).round(decimals= 2)
close_index = np.random.uniform(low= 50, high= 51, size= num_days)

volume_stock1 = np.random.uniform(low= 100, high= 150, size= num_days)
volume_stock2 = np.random.uniform(low= 1000, high= 1500, size= num_days)
volume_stocks = np.hstack((volume_stock1, volume_stock2)).round(decimals= 2)
volume_index = np.random.uniform(low= 2000, high= 2500, size= num_days)

turnover_stock1 = close_stock1 * volume_stock1
turnover_stock2 = close_stock2 * volume_stock2
turnover_stocks = np.hstack((turnover_stock1, turnover_stock2)).round(decimals= 2)
turnover_index = close_index * volume_index

high_stock1 = np.random.uniform(low= 11, high= 12, size= num_days)
high_stock2 = np.random.uniform(low= 21, high= 22, size= num_days)
high_stocks = np.hstack((high_stock1, high_stock2)).round(decimals= 2)
high_index = np.random.uniform(low= 51, high= 52, size= num_days)

low_stock1 = np.random.uniform(low= 9, high= 10, size= num_days)
low_stock2 = np.random.uniform(low= 19, high= 20, size= num_days)
low_stocks = np.hstack((low_stock1, low_stock2)).round(decimals= 2)
low_index = np.random.uniform(low= 49, high= 50, size= num_days)

open_stock1 = np.random.uniform(low= 10, high= 11, size= num_days)
open_stock2 = np.random.uniform(low= 20, high= 21, size= num_days)
open_stocks = np.hstack((open_stock1, open_stock2)).round(decimals= 2)
open_index = np.random.uniform(low= 50, high= 51, size= num_days)

df_stocks = pd.DataFrame({'order_book_id': stocks, 'datetime': datetime_stocks, 
                          'open': open_stocks, 'high': high_stocks,
                          'low': low_stocks, 'close': close_stocks,  
                          'volume': volume_stocks, 'total_turnover': turnover_stocks})
df_stocks.set_index(["order_book_id", "datetime"], inplace=True)
                          
df_index = pd.DataFrame({'order_book_id': index, 'datetime': datetime, 
                          'open': open_index, 'high': high_index,
                          'low': low_index, 'close': close_index,  
                          'volume': volume_index, 'total_turnover': turnover_index})
df_index.set_index(["order_book_id", "datetime"], inplace=True)




# ############### get stocks and index dataframes from jointquant, then store to pickle #########
# from jqdatasdk import *
# auth('17727977512', '147258369')

# stocks = ['000001.XSHE', '000002.XSHE']
# index = ['399006.XSHE']
# fields=['date','open','high','low','close','volume','money']

# stocks_100d = get_bars(stocks, fields= fields, unit= '1d', end_dt="2020-11-06", count=100)
# stocks_100d = stocks_100d.reset_index(level=1, drop=True).set_index("date", append=True)
# stocks_100d.index.rename(['order_book_id','datetime'], inplace=True)
# stocks_100d.rename(columns= {'money': 'total_turnover'}, inplace=True)
# stocks_100d = stocks_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))

# index_100d = get_bars(index, fields= fields, unit= '1d', end_dt="2020-11-06", count=100)
# index_100d = index_100d.reset_index(level=1, drop=True).set_index("date", append=True)
# index_100d.index.rename(['order_book_id','datetime'], inplace=True)
# index_100d.rename(columns= {'money': 'total_turnover'}, inplace=True)
# index_100d = index_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))


# ############## get input x and y tensors #############
# import featurizer.functors.journalhub as jf
# import get_securities_fields_tensors as gt
# import featurizer.functors.time_series as tf

# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')

# dict_index = gt.get_securities_fields_tensors(index_100d)
# dict_stocks = gt.get_securities_fields_tensors(stocks_100d)

# stocks_close_ts = dict_stocks['close']
# stocks_high_ts = dict_stocks['high']
# stocks_low_ts = dict_stocks['low']
# stocks_turnover_ts = dict_stocks['turnover']
# stocks_volume_ts = dict_stocks['volume']
# index_close_ts = dict_index['close']
# index_volume_ts = dict_index['volume']
# index_turnover_ts = dict_index['turnover']

# pct_change_functor = tf.PctChange()
# index_returns_ts = pct_change_functor(index_close_ts)
# stocks_returns_ts = pct_change_functor(stocks_close_ts)

# index_returns_ts_temp = index_returns_ts
# for i in range(0, list(stocks_returns_ts.size())[1] - 1):
#     index_returns_ts = torch.cat((index_returns_ts, index_returns_ts_temp), 1)
    
# stock_turnover_pct_ts = pct_change_functor(stocks_turnover_ts)
# stock_high_low_ratio_ts = torch.div(stocks_high_ts, stocks_low_ts)

# x_ts = torch.stack([stock_turnover_pct_ts, stock_high_low_ratio_ts, index_returns_ts])
# x_ts = x_ts.permute(2,1,0)
# y_ts = stocks_returns_ts.unsqueeze(-1).permute(1,0,2)


# ################## get expected output tensors ########
# ResidualRollingMeanFunctor = jf.ResidualRollingMean(window_train=30, window_test=20, window=10)
# ResidualRollingMeanFactor = ResidualRollingMeanFunctor.forward(x_ts, y_ts)

# ResidualRollingWeightedMeanFunctor = jf.ResidualRollingWeightedMean(window_train=30, window_test=20, window=10)
# ResidualRollingWeightedMeanFactor = ResidualRollingWeightedMeanFunctor.forward(x_ts, y_ts)

# ResidualRollingStdFunctor = jf.ResidualRollingStd(window_train=30, window_test=20, window=10)
# ResidualRollingStdFactor = ResidualRollingStdFunctor.forward(x_ts, y_ts)

# ResidualRollingWeightedStdFunctor = jf.ResidualRollingWeightedStd(window_train=30, window_test=20, window=10)
# ResidualRollingWeightedStdFactor = ResidualRollingWeightedStdFunctor.forward(x_ts, y_ts)

# ResidualRollingDownsideStdFunctor = jf.ResidualRollingDownsideStd(window_train=30, window_test=20, window=10)
# ResidualRollingDownsideStdFactor = ResidualRollingDownsideStdFunctor.forward(x_ts, y_ts)

# ResidualRollingMaxFunctor = jf.ResidualRollingMax(window_train=30, window_test=20, window=10)
# ResidualRollingMaxFactor = ResidualRollingMaxFunctor.forward(x_ts, y_ts)

# ResidualRollingMinFunctor = jf.ResidualRollingMin(window_train=30, window_test=20, window=10)
# ResidualRollingMinFactor = ResidualRollingMinFunctor.forward(x_ts, y_ts)

# ResidualRollingMedianFunctor = jf.ResidualRollingMedian(window_train=30, window_test=20, window=10)
# ResidualRollingMedianFactor = ResidualRollingMedianFunctor.forward(x_ts, y_ts)

# ResidualRollingSkewFunctor = jf.ResidualRollingSkew(window_train=30, window_test=20, window=10)
# ResidualRollingSkewFactor = ResidualRollingSkewFunctor.forward(x_ts, y_ts)

# ResidualRollingKurtFunctor = jf.ResidualRollingKurt(window_train=30, window_test=20, window=10)
# ResidualRollingKurtFactor = ResidualRollingKurtFunctor.forward(x_ts, y_ts)

# ResidualRollingCumulationFunctor = jf.ResidualRollingCumulation(window_train=30, window_test=20, window=10)
# ResidualRollingCumulationFactor = ResidualRollingCumulationFunctor.forward(x_ts, y_ts)

# ResidualRollingVARMOMFunctor = jf.ResidualRollingVARMOM(window_train=30, window_test=20, window=10)
# ResidualRollingVARMOMFactor = ResidualRollingVARMOMFunctor.forward(x_ts, y_ts)

# ResidualRollingMaxDrawdownFromReturnsFunctor = jf.ResidualRollingMaxDrawdownFromReturns(window_train=30, window_test=20, window=10)
# ResidualRollingMaxDrawdownFromReturnsFactor = ResidualRollingMaxDrawdownFromReturnsFunctor.forward(x_ts, y_ts)


# ################### Store all tensors #########

# t = {'x_ts': x_ts, 'y_ts': y_ts,
#      'ResidualRollingMeanFactor': ResidualRollingMeanFactor, 
#      'ResidualRollingWeightedMeanFactor': ResidualRollingWeightedMeanFactor,
#      'ResidualRollingStdFactor': ResidualRollingStdFactor, 
#      'ResidualRollingWeightedStdFactor': ResidualRollingWeightedStdFactor,
#      'ResidualRollingDownsideStdFactor': ResidualRollingDownsideStdFactor,
#      'ResidualRollingMaxFactor': ResidualRollingMaxFactor, 
#      'ResidualRollingMinFactor': ResidualRollingMinFactor,
#      'ResidualRollingMedianFactor': ResidualRollingMedianFactor,
#      'ResidualRollingSkewFactor': ResidualRollingSkewFactor,
#      'ResidualRollingKurtFactor': ResidualRollingKurtFactor,
#      'ResidualRollingCumulationFactor': ResidualRollingCumulationFactor,
#      'ResidualRollingVARMOMFactor': ResidualRollingVARMOMFactor,
#      'ResidualRollingMaxDrawdownFromReturnsFactor': ResidualRollingMaxDrawdownFromReturnsFactor
#      }

# torch.save(t, 'tensors_db.pt')




