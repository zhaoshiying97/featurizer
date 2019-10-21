#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import featurizer.functors as ff

np.random.seed(520)

def rolling_mean():
    window_size = 5
    
    data_df = pd.DataFrame(np.random.random((10,3)))
    output1 = data_df.rolling(window=window_size).mean()
    
    
    
    data_ts =  torch.tensor(data_df.values)
    functor = ff.RollingMean(window=window_size)
    output2 = functor(data_ts)
    
    # numpy: output1.values == output2.numpy()
    return output1, output2

if __name__ == "__main__":
    output1, output2 = rolling_mean()
    