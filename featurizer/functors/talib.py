#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pdb
import torch
from featurizer.interface import Functor
import featurizer.functions.time_series_functions as tsf
from featurizer.functions.talib import macd, rsi, bbands

class ROCP(Functor):
    def __init__(self, timeperiod=1):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        return tsf.pct_change(tensor)

class VolumeROCP(Functor):
    def __init__(self):
        pass
    def forward(self, tensor):
        tensor = torch.clamp(tensor, min=1)
        pct_change = tsf.pct_change(tensor)
        return torch.atan(pct_change)

class MAROCP(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod).squeeze(-1)
        return tsf.pct_change(tensor_rolling_mean)

class MARelative(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod).squeeze(-1)
        #tensor_rolling_mean = tensor_rolling_mean.squeeze(-1)
        relative =  (tensor_rolling_mean - tensor).div(tensor)
        return relative

class VolumeMAROCP(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod).squeeze(-1)
        volume_pct_change =  tsf.pct_change(tensor_rolling_mean)
        return torch.atan(volume_pct_change)
    
class VolumeRelative(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
        
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod)
        tensor_rolling_mean = tensor_rolling_mean.squeeze(-1)
        tensor = torch.clamp(tensor, min=1)
        relative =  (tensor_rolling_mean - tensor).div(tensor)
        return torch.atan(relative)

class MACDRelated(Functor):
    
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
    
    def forward(self, tensor):
        DIF, DEA, MACD = macd(tensor, fastperiod=self.fastperiod, slowperiod=self.slowperiod, signalperiod=self.signalperiod)
        #norm_macd = np.minimum(np.maximum(np.nan_to_num(macd), -1), 1)
        
        # norm
        norm_DIF = torch.clamp(DIF, min=-1, max=1).squeeze(-1)
        norm_DEA = torch.clamp(DEA, min=-1, max=1).squeeze(-1)
        norm_MACD = torch.clamp(MACD, min=-1, max=1).squeeze(-1)
        #pdb.set_trace()
        # diff
        norm_DIF_diff = torch.clamp(tsf.diff(DIF),min=-1, max=1).squeeze(-1)
        norm_DEA_diff = torch.clamp(tsf.diff(DEA),min=-1, max=1).squeeze(-1)
        norm_MACD_diff = torch.clamp(tsf.diff(MACD),min=-1, max=1).squeeze(-1)
        return norm_DIF, norm_DEA, norm_MACD, norm_DIF_diff, norm_DEA_diff, norm_MACD_diff

class DemeanedRSI(Functor):
    
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        rsi_ts = rsi(tensor, timeperiod=self.timeperiod).squeeze(-1)
        return rsi_ts/100 -0.5 

class RSIROCP(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        rsi_ts = rsi(tensor, timeperiod=self.timeperiod).squeeze(-1)
        rsi_pct_change = tsf.pct_change(rsi_ts+100)
        return rsi_pct_change
        

class BBANDS(Functor):
    def __init__(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        self.timeperiod = timeperiod
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.matype = matype
    
    def forward(self, tensor):
        upperband_ts, middleband_ts, lowerband_ts = bbands(tensor,timeperiod=self.timeperiod, nbdevup = self.nbdevup, nbdevdn=self.nbdevdn, matype=self.matype)
        #pdb.set_trace()
        upperband_relative_ts = (upperband_ts.squeeze(-1) - tensor)/tensor
        middleband_relative_ts = (middleband_ts.squeeze(-1) - tensor)/tensor
        lowerband_relative_ts = (lowerband_ts.squeeze(-1) - tensor)/tensor
        return upperband_relative_ts, middleband_relative_ts, lowerband_relative_ts  

class PriceVolume(Functor):
    def __init__(self):
        pass
    def forward(self, tensor_price, tensor_volume):
        rocp = tsf.pct_change(tensor_price)
        
        tensor_volume = torch.clamp(tensor_volume,min=1)
        v_pct_change = tsf.pct_change(tensor_volume)
        v_pct_change_atan = torch.atan(v_pct_change)

        pv = rocp * v_pct_change_atan
        return pv