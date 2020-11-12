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
import torch
import numpy as np


# ------ Load this code if not sure about the shape of input and output tensors
# all_tensors = torch.load('tensors_db.pt') 
# x_ts = all_tensors['x_ts']
# y_ts = all_tensors['y_ts']

############# create mock x, y, and residual tensors ##############
y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
y = torch.tensor(y)
y = y.unsqueeze(-2)
y = y.unsqueeze(-2)
y = y.permute(1,2,0)

x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x1 = torch.tensor(x1)
x1 = x1.unsqueeze(-2)
x2 = [10,20,30,40,50,60,70,80,90,100,
      110,120,130,140,150,160,170,180,190,200]
x2 = torch.tensor(x2)
x2 = x2.unsqueeze(-2)
x = torch.stack([x1,x2])
x = x.permute(1,2,0)

err = [0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2, 0.1, -0.1,
       0.2, -0.2, 0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2]
err = torch.tensor(err)
err = err.unsqueeze(-2)
err = err.unsqueeze(-2)
err = err.permute(1,2,0)

y = y + err

############### calculate expected factor tensors ##################
window_train, window_test, window = 10, 5, 3

from featurizer.functors.journalhub import *
std_functor = ResidualRollingStd(window_train= window_train, window_test= window_test, window= window)
std_factor = std_functor(tensor_x= x, tensor_y= y)

err_np = err.squeeze(0).cpu().detach().numpy()
err_df = pd.DataFrame(err_np)
std_df = err_df.rolling(window).std()


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




