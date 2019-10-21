#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from featurizer.functions.algebra_statistic import weighted_average, weighted_std, downside_std

# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def rolling_sum(tensor, window=1, dim=0):
    ret = torch.cumsum(tensor, dim=dim)
    ret[window:] = ret[window:] - ret[:-window]
    ret[:window-1]= float("nan")
    return ret

def rolling_sum_(tensor, window=1, dim=0):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).sum()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_sum3d(tensor, window=1, dim=1):
    ret = torch.cumsum(tensor, dim=dim)
    ret[:,window:] = ret[:,window:] - ret[:,:-window]
    ret[:,:window-1]= float("nan")
    return ret

def rolling_mean(tensor, window=1):
    ret = torch.cumsum(tensor, dim=0)
    ret[window:] = ret[window:] - ret[:-window]
    ret[:window-1]= float("nan")
    return ret/window

def rolling_mean_(tensor, window=1):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).mean()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_weighted_mean(tensor, window=1, halflife=90):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(lambda x: weighted_average(x, halflife=halflife))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(tensor, window=1):
    if window == 0:
        return tensor
    
    e = torch.empty_like(tensor, dtype=tensor.dtype, device=tensor.device)
    if window > 0:
        e[:window] = float("nan")
        e[window:] = tensor[:-window]
    else:
        e[window:] = float("nan")
        e[:window] = tensor[-window:]
    return e

def diff(tensor, period=1):
    shiftd_tensor = shift(tensor, window=period)
    diff = tensor - shiftd_tensor
    return diff

def pct_change(tensor, period=1):
    shiftd_tensor = shift(tensor, window=period)
    diff = tensor - shiftd_tensor
    output = diff.div(shiftd_tensor)
    return output 

#https://stackoverflow.com/questions/54564253/how-to-calculate-the-cumulative-product-of-a-rolling-window-in-pandas
def rolling_prod(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(np.prod)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_var(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).var()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor 

def rolling_std(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).std()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_weighted_std(tensor, window, halflife=90):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(lambda x: weighted_std(x, halflife=halflife))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_downside_std(tensor, tensor_benchmark, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_benchmark_np = tensor_benchmark.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    #tensor_benchmark_s = pd.Series(tensor_benchmark_np)
    output_df = tensor_df.rolling(window).apply(lambda x: downside_std(x, tensor_benchmark_np)) #fix me: shape error
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_skew(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).skew()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_max(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).max()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_cov(tensor_x, tensor_y, window, trailing_window=0):
    tensor_x_np = tensor_x.cpu().detach().numpy()
    tensor_y_np = tensor_y.cpu().detach().numpy()
    tensor_x_df = pd.DataFrame(tensor_x_np)
    tensor_y_df = pd.DataFrame(tensor_y_np)
    output_df = tensor_x_df.rolling(window).cov(tensor_y_df)
    output_tensor = torch.tensor(output_df.values, dtype=tensor_x.dtype, device=tensor_x.device)
    return output_tensor 

def ema(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.ewm(span=window,min_periods=window).mean()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

if __name__ == "__main__":
    torch.manual_seed(520)
    tensor_1d_float = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    tensor_2d_float = torch.randn(10,3)
    tensor_3d_float = torch.randn(2,10,3)
    
    tensor_2d_int = torch.randint(1,3,(10,3))
    tensor_3d_int = torch.randint(1,3,(2,10,3))
    
    tensor_1d_float_sum = rolling_sum(tensor_1d_float, window=2)
    tensor_2d_float_sum = rolling_sum(tensor_2d_float, window=2)
    tensor_3d_float_sim = rolling_sum3d(tensor_3d_float, window=2, dim=1)

#    output_ma = rolling_mean(a, window=2)
#    output1_shift1 = shift(a, window=1)
#    output1_shift2 = shift(a, window=2)
#    output1_shift_1 = shift(a, window=-1)
    
    #pct_change_1 = pct_change(a,period=1)
    #pct_change_2 = pct_change(b,period=1)
    
#    b = torch.randn(8,3)
#    output2_shift1 = shift(b, window=1)
#    output2_shift2 = shift(b, window=2)
#    output2_shift_1 = shift(b, window=-1)
    #rolling_var = rolling_var(b,window=3)
    a = torch.randn(8,3)
    b = torch.randn(8,3)
    
    rolling_cov = rolling_cov(a,b,window=3)
    rolling_var = rolling_var(a, window=3)
    #prod_result = rolling_prod(a, window=3)
    rolling_weighted_mean_value = rolling_weighted_mean(a, window=4)
    rolling_weighted_std_value = rolling_weighted_std(a, window=4)
    rolling_skew_value = rolling_skew(a, window=4)
    
    #rolling_downside_std_value = rolling_downside_std(a, a[:,0], window=4)
