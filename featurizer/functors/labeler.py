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

from featurizer.interface import Functor
from featurizer.functions import time_series_functions as tsf

class RegressionLabler(Functor):
    
    def __init__(self, window=1):
        self._window = window
        
    def forward(self, tensor):
        returns = tsf.pct_change(tensor, period=self._window)
        return tsf.shift(returns, window=-self._window)

if __name__ == "__main__":
    import torch
    close = abs(torch.randn(10,3))
    regression_funtor = RegressionLabler(window=2)
    label = regression_funtor.forward(close)