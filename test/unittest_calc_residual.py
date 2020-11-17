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
y_2d= y_fitted + randn(n)

# parameters calculated from built-in functions in numpy
A = sm.add_constant(x_2d)
result = np.linalg.lstsq(A, y_2d)
b,a1,a2 = result[0]

# 2d tensors
x_2d_ts = torch.tensor(x_2d)
y_2d_ts = torch.tensor(y_2d)

# 3d-x and 3d-y
x_3d_ts = torch.stack((x_2d_ts, x_2d_ts)) # x_3d_ts.shape == (2,20,3)

y_stack = torch.stack((y_2d_ts, y_2d_ts))
y_3d_ts = y_stack.unsqueeze(-1) # y_3d_ts.shape == (1,20,3)


# Begin tests
class Test_get_algebra_coef_ts(unittest.TestCase):
    
    expected_a1, expected_a2, expected_b = a1, a2, b
    
    def test_forward(self):
        self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[0,0,0] - b < 0.001)
        self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[1,0,0] - b < 0.001)
        self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[0,1,0] - a1 < 0.001)
        self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[1,1,0] - a1 < 0.001)
        self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[0,2,0] - a2 < 0.001)
        self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[1,2,0] - a2 < 0.001)
        
      
# class Test_get_residual_ts(unittest.TestCase):
    
#     expected_a1, expected_a2, expected_b = a1, a2, b
    
#     def test_forward(self):
#         self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[0,0,0] - b < 0.001)
#         self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[1,0,0] - b < 0.001)
#         self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[0,1,0] - a1 < 0.001)
#         self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[1,1,0] - a1 < 0.001)
#         self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[0,2,0] - a2 < 0.001)
#         self.assertTrue(get_algebra_coef_ts(x_3d_ts, y_3d_ts)[1,2,0] - a2 < 0.001)      


if __name__ =='__main__':
    unittest.main()
    
    
    