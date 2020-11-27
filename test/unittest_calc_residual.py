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
    
        # ----------- Create test data required for all subsequent tests ----------------
        np.random.seed(555)
        
        self.n = int(20) #5e6
        x1 = np.linspace(-10,10,self.n)
        x2 = np.linspace(-5,3,self.n)
        self.x_2d = np.column_stack((x1, x2))
        
        # parameters:
        # a is coefficient and b is intercept
        a1=3.25; a2=2; b=-6.5
        y_fitted = a1*x1 + a2*x2 + b
        # add some noise
        self.y = y_fitted + randn(self.n)
        self.y_2d = np.expand_dims(self.y, axis= 1)
        
        # # parameters calculated from built-in functions in numpy
        # self.A = sm.add_constant(self.x_2d)
        # self.expected_params = np.linalg.lstsq(self.A, self.y)
        # # expected_b, expected_a1, expected_a2 = expected_params[0]
        # self.err = np.sqrt(self.expected_params[1]/len(self.y))
        
        # parameters calculated from built-in functions in Statsmodels.OLS
        self.A = sm.add_constant(self.x_2d)
        model = sm.OLS(self.y_2d, self.A)
        results = model.fit().params
        self.expected_params = np.expand_dims(results, axis=-1) # (3,1)
        
        # 2d tensors
        self.y_ts = torch.tensor(self.y)
        self.y_2d_ts = torch.stack((self.y_ts, self.y_ts))
        self.x_2d_ts = torch.tensor(self.x_2d)
        
        # 3d-x and 3d-y
        self.x_3d_ts = torch.stack((self.x_2d_ts, self.x_2d_ts)) # x_3d_ts.shape == (2,20,2)
        self.y_3d_ts = self.y_2d_ts.unsqueeze(-1) # y_3d_ts.shape == (1,20,2)
        
        self.error_decimal_threshold = 5
    
    
    def test_get_algebra_coef_ts(self):
        
        expected_b, expected_a1, expected_a2 = self.expected_params[:,0]
        
        # actual output
        output_coef = get_algebra_coef_ts(self.x_3d_ts, self.y_3d_ts)
        
        # check if the actual and expected parameters are almost equal
        self.assertAlmostEqual(output_coef[0,0,0].item(), expected_b, self.error_decimal_threshold, 
                               'Expected param b is different from the actual for 1st commpany')
        self.assertAlmostEqual(output_coef[1,0,0].item(), expected_b, self.error_decimal_threshold,
                               'Expected param b is different from the actual for 2nd commpany')
        self.assertAlmostEqual(output_coef[0,1,0].item(), expected_a1, self.error_decimal_threshold, 
                               'Expected param a1 is different from the actual for 1st commpany')
        self.assertAlmostEqual(output_coef[1,1,0].item(), expected_a1, self.error_decimal_threshold, 
                               'Expected param a1 is different from the actual for 2nd commpany')
        self.assertAlmostEqual(output_coef[0,2,0].item(), expected_a2, self.error_decimal_threshold,
                               'Expected param a2 is different from the actual for 1st commpany')
        self.assertAlmostEqual(output_coef[1,2,0].item(), expected_a2, self.error_decimal_threshold,
                               'Expected param a2 is different from the actual for 2nd commpany')
        
        
    # def test_get_algebra_coef_ts_partial(self):
    #     y_partial = self.y_3d_ts[0][8:-7]
    #     x_partial = self.x_3d_ts[0][8:-7]
    #     A_partial = sm.add_constant(x_partial)
    #     expected_params_partial = np.linalg.lstsq(A_partial, y_partial)[0]
        
    #     x_partial_3d = torch.stack((x_partial, x_partial))
    #     y_partial_3d = torch.stack((y_partial, y_partial))
    #     output_params_partial = get_algebra_coef_ts(x_partial_3d, y_partial_3d)[0].numpy()
    #     diff = expected_params_partial - output_params_partial
        
    #     self.assertTrue((expected_params_partial == output_params_partial).all(), diff)
    
    
    def test_get_residual_ts(self):
        # pdb.set_trace()
        # get expected parameters and expected residuals in the desired format
        # expected_params_2d = np.expand_dims(self.expected_params[0],axis=0).T
        expected_params_3d_half = np.expand_dims(self.expected_params, axis=0)
        expected_params_3d = np.vstack((expected_params_3d_half, expected_params_3d_half))
        expected_residuals_2d = np.expand_dims(self.y, axis= 1) - self.A @ self.expected_params
        
        # actual output
        output_residuals = get_residual_ts(self.x_3d_ts, self.y_3d_ts, expected_params_3d)
        
        # check if the difference between the summation of the actual and the expected residuals is almost zero
        #  - print the difference if failed
        diff_2d = output_residuals[0,:,:] - expected_residuals_2d
        error_threshold = 0.1 * self.error_decimal_threshold
        self.assertTrue(abs((output_residuals[0,:,:] - expected_residuals_2d).sum()) < error_threshold, diff_2d)
    
    
    # helper function: recursively get expected residuals for the case of rolling
    def get_expected_rolling_resid(self, x_2dnp_with_constant, y, window_train, window_test, n, keep_first_train_nan):
        pdb.set_trace()
        if not n % window_train: 
            num_rolling = n // window_train
        else: # The last group of data for test is less than window_test
            num_rolling = n // window_train + 1
        # num_rolling = n // window_train - 1
        
        # Initialize the expected residuals differently based on whether to keep first train NaN or not
        if keep_first_train_nan:
            expected_resid = np.expand_dims(np.array([np.nan] * window_train), axis=0).T
        else:
            cur_params = sm.OLS(x_2dnp_with_constant[:window_train, :], y[:window_train]).fit().params.T
            # cur_params = np.expand_dims(cur_params, axis=-1) # (3,1)
            expected_resid = np.expand_dims(y[:window_train] - x_2dnp_with_constant[:window_train, :] @ cur_params, axis=1)
        
        # Rercursively get expected residuals
        for i in range(num_rolling):
            # print('round', i)
            # pdb.set_trace()
            if i == num_rolling - 1 and (n-window_train) % window_test: # last round of rolling, and when the size of data for test is less than window_test
                test_size = n - window_train - window_test * (num_rolling-1)
                y_train = y[n-test_size-window_train : test_size*-1]
                y_test = y[test_size*-1 : ]
                x_train = x_2dnp_with_constant[n-test_size-window_train : test_size*-1, :]
                x_test = x_2dnp_with_constant[test_size*-1 : ]
                # print('test_size:', test_size)
            else:
                y_train = y[i*window_test : i*window_test+window_train]
                y_test = y[i*window_test+window_train : (i+1)*window_test+window_train]
                x_train = x_2dnp_with_constant[i*window_test : i*window_test+window_train, :]
                x_test = x_2dnp_with_constant[i*window_test+window_train : (i+1)*window_test+window_train, :]
                # print('test_size:', window_test)
            
            cur_params = sm.OLS(x_train, y_train).fit().params.T
            # cur_params = np.expand_dims(cur_params, axis=-1) # (3,1)
            y_test_2d = np.expand_dims(y_test, axis= 1)
            cur_resid = y_test_2d - x_test @ cur_params
            expected_resid = np.vstack((expected_resid, cur_resid)) # shape == (n,1)
            # print('y_train:', y_train)
            # print('y_test:', y_test)
            # print('cur_params:\n', cur_params)
            # print('cur_resid:\n', cur_resid)
            # print('expected_resid:\n', expected_resid, '\n\n')
        return expected_resid
        
        
    # def test_calc_residual3d_ts_first_train_NaN(self):
        
    #     window_train, window_test = 5, 5
    #     '''
    #     manually make expected 2d residuals in numpy recursively
    #        - The first 5 entries should be NaN
    #        - Rolling should occur 4 times; Test window sizes are 5, 5, 5, 5, respectively
    #     '''
    #     expected_resid = self.get_expected_rolling_resid(self.A, self.y, window_train, window_test, self.n, True)
    #     expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
    #     expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
    #     output_resid = calc_residual3d_ts(self.x_3d_ts, self.y_3d_ts, window_train=window_train, 
    #                                       window_test=window_test, keep_first_train_nan= True)
    #     output_resid_np = np.array(output_resid)
        
    #     # Check if the difference between the expected and actual residuals sum up to almost 0
    #     #   - print the difference if failed
    #     err_threshold = 0.1 * self.error_decimal_threshold
    #     diff = (output_resid_np - expected_resid_3d).round(3)
    #     self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
        
    # def test_calc_residual3d_ts_first_train_not_NaN(self):
        
        # window_train, window_test = 10, 5
        # '''
        # manually make expected 2d residuals in numpy recursively
        #    - The first 5 entries should NOT be NaN
        #    - Rolling should occur 4 times; Test window sizes are 5, 5, 5, 5, respectively
        # '''
        # expected_resid = self.get_expected_rolling_resid(self.A, self.y, window_train, window_test, self.n, False)
        # expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        # expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
    #     output_resid = calc_residual3d_ts(self.x_3d_ts, self.y_3d_ts, window_train=window_train, 
    #                                       window_test=window_test, keep_first_train_nan= False)
    #     output_resid_np = np.array(output_resid)
        
    #     # Check if the difference between the expected and actual residuals sum up to almost 0
    #     #   - print the difference if failed
    #     err_threshold = 0.00001 * self.error_decimal_threshold
    #     diff = (output_resid_np - expected_resid_3d).round(3)
    #     self.assertTrue(abs(np.sum(diff)) < err_threshold, diff)
        
    
    # '''
    # This function currently is expected to fail. 
    #     - The difference should be between the residuals from the last two rollings.
    #     - Bug possibly from function split_sample3d

    # '''
    def test_calc_residual3d_ts_irregular_last_test_size(self):
        
        window_train, window_test = 5, 4
        '''
        manually make expected 2d residuals in numpy recursively
           - The first 9 entries should be NaN
           - Rolling should occur 3 times; Test window sizes are 5, 5, 1, respectively
        '''
        # pdb.set_trace()
        expected_resid = self.get_expected_rolling_resid(self.A, self.y, window_train, window_test, self.n, True)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid = calc_residual3d_ts(self.x_3d_ts, self.y_3d_ts, window_train=window_train, 
                                          window_test=window_test, keep_first_train_nan= True)
        output_resid_np = output_resid.numpy()
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
     

if __name__ =='__main__':
    unittest.main()
    
    
    