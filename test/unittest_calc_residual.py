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
    
    def setUp(self):
    
    # ----------- Create a data in 3d tensors ----------------
        np.random.seed(555)
        
        self.n=int(20) #5e6
        x1=np.linspace(-10,10,self.n)
        x2=np.linspace(-5,3,self.n)
        self.x_2d = np.column_stack((x1, x2))
        
        # parameters:
        # a is coefficient and b is intercept
        a1=3.25; a2=2; b=-6.5
        y_fitted = a1*x1 + a2*x2 + b
        # add some noise
        self.y = y_fitted + randn(self.n)
        # self.y_2d = np.expand_dims(self.y, axis= 1)
        
        # parameters calculated from built-in functions in numpy
        self.A = sm.add_constant(self.x_2d)
        self.expected_params = np.linalg.lstsq(self.A, self.y)
        # expected_b, expected_a1, expected_a2 = expected_params[0]
        self.err = np.sqrt(self.expected_params[1]/len(self.y))
        
        # 2d tensors
        self.y_ts = torch.tensor(self.y)
        self.y_2d_ts = torch.stack((self.y_ts, self.y_ts))
        self.x_2d_ts = torch.tensor(self.x_2d)
        
        # 3d-x and 3d-y
        self.x_3d_ts = torch.stack((self.x_2d_ts, self.x_2d_ts)) # x_3d_ts.shape == (2,20,2)
        self.y_3d_ts = self.y_2d_ts.unsqueeze(-1) # y_3d_ts.shape == (1,20,2)
    
    def test_get_algebra_coef_ts(self):
        
        expected_b, expected_a1, expected_a2 = self.expected_params[0]
        
        output_coef = get_algebra_coef_ts(self.x_3d_ts, self.y_3d_ts)
        
        # check if the difference between the actual and expected parameters are zero
        self.assertTrue(abs(output_coef[0,0,0] - expected_b) < 0.001)
        self.assertTrue(abs(output_coef[1,0,0] - expected_b) < 0.001)
        self.assertTrue(abs(output_coef[0,1,0] - expected_a1) < 0.001)
        self.assertTrue(abs(output_coef[1,1,0] - expected_a1) < 0.001)
        self.assertTrue(abs(output_coef[0,2,0] - expected_a2) < 0.001)
        self.assertTrue(abs(output_coef[1,2,0] - expected_a2) < 0.001)
    
    
    def test_get_residual_ts(self):
        
        # expected data for testing residuals
        expected_params_2d = np.expand_dims(self.expected_params[0],axis=0).T
        expected_params_3d_half = np.expand_dims(expected_params_2d, axis=0)
        expected_params_3d = np.vstack((expected_params_3d_half, expected_params_3d_half))
        expected_residuals_2d = np.expand_dims(self.y, axis= 1) - self.A @ expected_params_2d
        
        output_residuals = get_residual_ts(self.x_3d_ts, self.y_3d_ts, expected_params_3d)
        diff_2d = output_residuals[0,:,:] - expected_residuals_2d
        self.assertTrue(diff_2d.sum() < 0.0000001)
        
    # def test_calc_residual3d_ts(self):
        
        
     

if __name__ =='__main__':
    unittest.main()
    
    
    