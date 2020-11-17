#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:43:00 2020

@author: wanghuanqiu
"""


import unittest
import torch
from featurizer.functions.calc_residual import *
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


class TestOLSMethods(unittest.TestCase):
    
    # ----------- Create data in 3d tensors ----------------
    np.random.seed(555)
    
    n=int(20) #5e6
    x1=np.linspace(-10,10,n)
    x2=np.linspace(-5,3,n)
    x_2d = np.column_stack((x1, x2))
    
    # parameters:
    # a is coefficient and b is intercept
    a1=3.25; a2=2; b=-6.5
    y_fitted = a1*x1 + a2*x2 + b
    # add some noise
    y = y_fitted + randn(n)
    y_2d = np.expand_dims(y, axis= 1)
    
    # parameters calculated from built-in functions in numpy
    A = sm.add_constant(x_2d)
    result = np.linalg.lstsq(A, y)
    expected_b, expected_a1, expected_a2 = result[0]
    err = np.sqrt(result[1]/len(y))
    
    # 2d tensors
    y_ts = torch.tensor(y)
    y_2d_ts = torch.stack((y_ts, y_ts))
    x_2d_ts = torch.tensor(x_2d)
    
    # 3d-x and 3d-y
    x_3d_ts = torch.stack((x_2d_ts, x_2d_ts)) # x_3d_ts.shape == (2,20,3)
    y_3d_ts = y_2d_ts.unsqueeze(-1) # y_3d_ts.shape == (1,20,3)
    
    def test_get_algebra_coef_ts(self):
        
        output_coef = get_algebra_coef_ts(self.x_3d_ts, self.y_3d_ts)
        self.assertTrue(abs(output_coef[0,0,0] - self.expected_b) < 0.001)
        self.assertTrue(abs(output_coef[1,0,0] - self.expected_b) < 0.001)
        self.assertTrue(abs(output_coef[0,1,0] - self.expected_a1) < 0.001)
        self.assertTrue(abs(output_coef[1,1,0] - self.expected_a1) < 0.001)
        self.assertTrue(abs(output_coef[0,2,0] - self.expected_a2) < 0.001)
        self.assertTrue(abs(output_coef[1,2,0] - self.expected_a2) < 0.001)
    
    # expected data for testing residuals
    expected_param_2d = np.expand_dims(result[0],axis=0).T
    expected_param_3d_half = np.expand_dims(expected_param_2d, axis=0)
    expected_param = np.vstack((expected_param_3d_half, expected_param_3d_half))
    expected_residuals_2d = y_2d - A @ expected_param_2d
    
    def test_get_residual_ts(self):
        
        output_residuals = get_residual_ts(self.x_3d_ts, self.y_3d_ts, self.expected_param)
        diff_2d = output_residuals[0,:,:] - self.expected_residuals_2d
        self.assertTrue(diff_2d.sum() < 0.0000001)
     

if __name__ =='__main__':
    unittest.main()
    
    
    