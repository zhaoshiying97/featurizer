#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:00:40 2020

@author: wanghuanqiu
"""

import numpy as np
import pandas as pd
import torch


# Input a securities dataframe with fields 'open','close','high','low','volume','total_turnover'
# and output a dictionary, of which the keys are the above strings and the values are corresponding tensors
def get_securities_fields_tensors(securities: pd.DataFrame) -> dict :  

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # --- get "high" tensor
    high = securities.drop(['open','close','low','volume','total_turnover'], axis= 1).unstack(0)
    high = torch.tensor(high.values, device= device)
    
    # --- get "low" tensor
    low = securities.drop(['open','close','high','volume','total_turnover'], axis= 1).unstack(0)
    low = torch.tensor(low.values, device= device)
    
    # --- get "close" tensor
    close = securities.drop(['open','high','low','volume','total_turnover'], axis= 1).unstack(0)
    close = torch.tensor(close.values, device= device)
    
    # --- get "turnover" tensor
    turnover = securities.drop(['open','close','high','volume','low'], axis= 1).unstack(0)
    turnover = torch.tensor(turnover.values, device= device)
    
    # --- get "volume" tensor
    volume = securities.drop(['open','close','high','total_turnover','low'], axis= 1).unstack(0)
    volume = torch.tensor(volume.values, device= device)
    
    # --- get "open" tensor
    opening = securities.drop(['close','high','low','volume','total_turnover'], axis= 1)
    opening = torch.tensor(opening.unstack(0).values)
    
    d = {'open': opening, 'high': high, 'low': low, 'close': close, 'volume': volume, 'turnover': turnover}
    
    return d
    
    

if __name__ == '__main__':
    
    # --- get data in the form of dataframe   
    from simons.api import history_bars

    stocks = ['000001.XSHE', '000002.XSHE']
    index = ['399006.XSHE']
    
    dt = pd.Timestamp("2020-10-22 15:30:00")
    fields=['datetime','open','high','low','close','volume','total_turnover']
    stocks_100d = history_bars(order_book_ids= stocks, dt= dt, bar_count= 100, frequency= '1d', fields= fields)
    index_100d = history_bars(order_book_ids= index, dt= dt, bar_count= 100, frequency= '1d', fields= fields)
    stocks_100d = stocks_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))
    index_100d = index_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))


    di = get_securities_fields_tensors(index_100d)
    ds = get_securities_fields_tensors(stocks_100d)