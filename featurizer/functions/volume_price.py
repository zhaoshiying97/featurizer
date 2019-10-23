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

import torch
import pandas as pd
import featurizer.functions.time_series_functions as tsf


def vwap(tensor_x, tensor_y, window):
    tensor_x_np = tensor_x.cpu().detach().numpy()
    tensor_y_np = tensor_y.cpu().detach().numpy()
    tensor_x_df = pd.DataFrame(tensor_x_np)
    tensor_y_df = pd.DataFrame(tensor_y_np)
    
    vwap_df = (tensor_x_df * tensor_y_df).rolling(window).sum() / tensor_y_df.rolling(window).sum()
    vwap_ts = torch.tensor(vwap_df.values, dtype=tensor_x.dtype, device=tensor_x.device)
    return vwap_ts