#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from featurizer.interface import Functor
from featurizer.functions import time_series_functions as tsf

class RollingMean(Functor):

    def __init__(self, window):
        super(RollingMean, self).__init__()
        self._window = window

    def forward(self, tensor):
        return tsf.rolling_mean(tensor, window=self._window)

class RollingWeightedMean(Functor):
    
    def __init__(self, window, halflife):
        self._window = window
        self._halflife = halflife
    
    def forward(self, tensor):
        return tsf.rolling_weighted_mean(tensor, window=self._window, halflife=self._halflife)

class RollingStd(Functor):
    
    def __init__(self, window):
        super(RollingStd, self).__init__()
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_std(tensor, window=self._window)

class RollingWeightedStd(Functor):
    
    def __init__(self, window, halflife):
        self._window = window
        self._halflife = halflife
    
    def forward(self, tensor):
        return tsf.rolling_weighted_std(tensor, window=self._window, halflife=self._halflife)

class RollingMeanScaledByStd(Functor):
    
    def __init__(self, window):
        super(RollingMeanScaledByStd, self).__init__()
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_mean(tensor, window=self._window) / tsf.rolling_std(tensor, window=self._window) 

class RollingSkew(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_skew(tensor, window=self._window)   

class PctChange(Functor):

    def __init__(self, window=1):
        super(PctChange, self).__init__()
        self._window = window

    def forward(self, tensor):
        return tsf.pct_change(tensor, period=self._window)

class RollingMax(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_max(tensor, window=self._window)

if __name__ == "__main__":
    import torch
    
    input_data = torch.randn(20, 10)
    input_close = abs(input_data)
    
    rolling_mean_func = RollingMean(window=5)
    output1 = rolling_mean_func(input_data)