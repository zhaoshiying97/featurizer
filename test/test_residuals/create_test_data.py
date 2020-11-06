# -*- coding: utf-8 -*-

# Author: Huanqiu Wang
# Created on: Nov 6, 2020

'''
This module contains the code that creates data for test cases, including
    - data needed while running the unit tests
    - the expected output data to be compared with dynamic outputs
'''

import pandas as pd
import pickle
import torch

############### get stocks and index dataframes from jointquant
from jqdatasdk import *
auth('17727977512', '147258369')

stocks = ['000001.XSHE', '000002.XSHE']
index = ['399006.XSHE']
fields=['date','open','high','low','close','volume','money']

stocks_100d = get_bars(stocks, fields= fields, unit= '1d', end_dt="2020-11-06", count=100)
stocks_100d = stocks_100d.reset_index(level=1, drop=True).set_index("date", append=True)
stocks_100d.index.rename(['order_book_id','datetime'], inplace=True)
stocks_100d.rename(columns= {'money': 'total_turnover'}, inplace=True)
stocks_100d = stocks_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))

index_100d = get_bars(index, fields= fields, unit= '1d', end_dt="2020-11-06", count=100)
index_100d = index_100d.reset_index(level=1, drop=True).set_index("date", append=True)
index_100d.index.rename(['order_book_id','datetime'], inplace=True)
index_100d.rename(columns= {'money': 'total_turnover'}, inplace=True)
index_100d = index_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))

stocks_100d.to_pickle('stocks_100d.pkl')
index_100d.to_pickle('index_100d.pkl')