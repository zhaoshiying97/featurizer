# -*- coding: utf-8 -*-
# Copyright StateOfTheArt.quant. 
#
# * Commercial Usage: please contact allen.across@gmail.com
# * Non-Commercial Usage:
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

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

def kdj(high_ts, low_ts, close_ts, fastk_period=9, slowk_period=3, slowd_period=3):
    range_ts = tsf.rolling_max(high_ts, window=fastk_period) - tsf.rolling_min(low_ts, window=fastk_period)
    RSV = (close_ts - tsf.rolling_min(low_ts, fastk_period).squeeze(-1)) / torch.clamp(range_ts.squeeze(-1),min=1)
    K = tsf.ema(RSV, window=slowk_period).squeeze(-1) 
    D = tsf.ema(K, slowd_period).squeeze(-1)
    J = 3*K - 2*D
    return RSV, K, D, J
