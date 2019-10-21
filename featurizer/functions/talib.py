#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pdb
import torch
import pandas as pd
import featurizer.functions.time_series_functions as tsf

def macd(tensor, fastperiod=12, slowperiod=26, signalperiod=9):
    #DIF = tsf.ema(tensor, fastperiod) - tsf.ema(tensor, slowperiod)
    #DEA = tsf.ema(DIF, signalperiod)
    #MACD = (DIF - DEA) * 1 # Here is 1 rather than trodational 2
    import talib
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    DIF = tensor_df.apply(lambda x: talib.MACD(x, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0])
    DEA = tensor_df.apply(lambda x: talib.MACD(x, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[1])
    MACD = tensor_df.apply(lambda x: talib.MACD(x, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[2])
    
    DIF_ts = torch.tensor(DIF.values, dtype=tensor.dtype, device=tensor.device)
    DEA_ts = torch.tensor(DEA.values, dtype=tensor.dtype, device=tensor.device)
    MACD_ts = torch.tensor(MACD.values, dtype=tensor.dtype, device=tensor.device)
    #
    return DIF_ts, DEA_ts, MACD_ts

    return DIF, DEA, MACD


def rsi(tensor, timeperiod):
    import talib
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    tensor_df = tensor_df.apply(lambda x: talib.RSI(x, timeperiod=timeperiod))
    output_tensor = torch.tensor(tensor_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor


def bbands(tensor, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    import talib
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    upperband = tensor_df.apply(lambda x: talib.BBANDS(x, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[0])
    middleband = tensor_df.apply(lambda x: talib.BBANDS(x, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[1])
    lowerband = tensor_df.apply(lambda x: talib.BBANDS(x, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[2])
    
    upperband_ts = torch.tensor(upperband.values, dtype=tensor.dtype, device=tensor.device)
    middleband_ts = torch.tensor(middleband.values, dtype=tensor.dtype, device=tensor.device)
    lowerband_ts = torch.tensor(lowerband.values, dtype=tensor.dtype, device=tensor.device)
    return upperband_ts, middleband_ts, lowerband_ts       
    