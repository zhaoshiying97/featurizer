#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:13:06 2020

@author: wanghuanqiu
"""

import unittest
import torch
from featurizer.functors.journalhub import *


# get tensors needed for the test cases
all_tensors = torch.load('tensors_db.pt') 
x_ts = all_tensors['x_ts']
y_ts = all_tensors['y_ts']
window_train, window_test, window = 30, 20, 10



class TestResidualRollingMean(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingMeanFactor']
    ResidualRollingMeanFunctor = ResidualRollingMean(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingMeanFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingWeightedMean(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingWeightedMeanFactor']
    ResidualRollingWeightedMeanFunctor = ResidualRollingWeightedMean(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingWeightedMeanFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))


class TestResidualRollingStd(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingStdFactor']
    ResidualRollingStdFunctor = ResidualRollingStd(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingStdFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingWeightedStd(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingWeightedStdFactor']
    ResidualRollingWeightedStdFunctor = ResidualRollingWeightedStd(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingWeightedStdFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        

class TestResidualRollingDownsideStd(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingDownsideStdFactor']
    ResidualRollingDownsideStdFunctor = ResidualRollingDownsideStd(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingDownsideStdFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingMax(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingMaxFactor']
    ResidualRollingMaxFunctor = ResidualRollingMax(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingMaxFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        

class TestResidualRollingMin(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingMinFactor']
    ResidualRollingMinFunctor = ResidualRollingMin(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingMinFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingMedian(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingMedianFactor']
    ResidualRollingMedianFunctor = ResidualRollingMedian(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingMedianFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingSkew(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingSkewFactor']
    ResidualRollingSkewFunctor = ResidualRollingSkew(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingSkewFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingKurt(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingKurtFactor']
    ResidualRollingKurtFunctor = ResidualRollingKurt(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingKurtFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingCumulation(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingCumulationFactor']
    ResidualRollingCumulationFunctor = ResidualRollingCumulation(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingCumulationFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingVARMOM(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingVARMOMFactor']
    ResidualRollingVARMOMFunctor = ResidualRollingVARMOM(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingVARMOMFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
        
        
class TestResidualRollingMaxDrawdownFromReturns(unittest.TestCase):
    
    expected_output = all_tensors['ResidualRollingMaxDrawdownFromReturnsFactor']
    ResidualRollingMaxDrawdownFromReturnsFunctor = ResidualRollingMaxDrawdownFromReturns(window_train, window_test, window)
    
    def test_forward(self):
        self.assertTrue(torch.allclose(self.ResidualRollingMaxDrawdownFromReturnsFunctor.forward(x_ts, y_ts), self.expected_output, equal_nan=True))
      
        


if __name__ =='__main__':
    unittest.main()