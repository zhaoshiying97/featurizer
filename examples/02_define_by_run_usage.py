#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from featurizer.interface import Functor
import featurizer.functors as ff


class Feature(Functor):
    
    def __init__(self):
        self.rolling_mean = ff.RollingMean(window=5)
        self.rolling_std = ff.RollingStd(window=3)
    
    def forward(self, x):
        feature1 = self.rolling_mean(x)
        feature2 = self.rolling_std(x)
        
        return [feature1, feature2]

if __name__ == "__main__":
    import torch
    data = torch.randn((10,3))
    
    my_functor = Feature()
    feature_list = my_functor(data)