import unittest

import numpy as np
import torch

import featurizer.functions.time_series_functions as tsf


class TestnewTsf(unittest.TestCase):

    def setUp(self):
        mock_rising = torch.range(1, 5, 1)
        mock_falling = torch.range(5, 1, -1)
        mock_ones = torch.ones(5)
        mock_twos = 2 * torch.ones(5)
        self.test_2d_1big = torch.stack([mock_twos, mock_ones], dim=1)
        self.test_2d_2big = torch.stack([mock_ones, mock_twos], dim=1)
        self.test_2d_ties = torch.stack([mock_ones, mock_ones], dim=1)
        self.test_2d_change = torch.stack([mock_rising, mock_falling], dim=1)

    def test_rank_1big(self):
        output_rank = tsf.rank(self.test_2d_1big)
        expected = torch.tensor([[1, 0.5], [1, 0.5], [1, 0.5], [1, 0.5], [1, 0.5]])
        result = (expected == output_rank).all().item()
        self.assertTrue(result, 'test_rank_1big failed')

    def test_rank_2big(self):
        output_rank = tsf.rank(self.test_2d_2big)
        expected = torch.tensor([[0.5, 1], [0.5, 1], [0.5, 1], [0.5, 1], [0.5, 1]])
        result = (expected == output_rank).all().item()
        self.assertTrue(result, 'test_rank_2big failed')

    def test_rank_ties(self):
        output_rank = tsf.rank(self.test_2d_ties)
        expected = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        result = (expected == output_rank).all().item()
        self.assertTrue(result, 'test_rank_ties failed')

    def test_tsrank_change(self):
        output_tsrank = tsf.ts_rank(self.test_2d_change, 2)
        output_tsrank = torch.where(torch.isnan(output_tsrank), torch.full_like(output_tsrank, 666),
                                    output_tsrank)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [2, 1], [2, 1], [2, 1], [2, 1]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)  # fillna
        result = (expected == output_tsrank).all().item()
        self.assertTrue(result, 'test_tsrank_change failed')

    def test_tsrank_ties(self):
        output_tsrank = tsf.ts_rank(self.test_2d_1big, 2)
        output_tsrank = torch.where(torch.isnan(output_tsrank), torch.full_like(output_tsrank, 666),
                                    output_tsrank)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [1, 1], [1, 1], [1, 1], [1, 1]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)  # fillna
        result = (expected == output_tsrank).all().item()
        self.assertTrue(result, 'test_tsrank_ties failed')

    def test_argmax_change(self):
        output_tsargmax = tsf.ts_argmax(self.test_2d_change, 2)
        output_tsargmax = torch.where(torch.isnan(output_tsargmax), torch.full_like(output_tsargmax, 666),
                                      output_tsargmax)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [1, 0], [1, 0], [1, 0], [1, 0]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)  # fillna
        result = (expected == output_tsargmax).all().item()
        self.assertTrue(result, 'test_argmax_change failed')

    def test_argmax_ties(self):
        output_tsargmax = tsf.ts_argmax(self.test_2d_ties, 2)
        output_tsargmax = torch.where(torch.isnan(output_tsargmax), torch.full_like(output_tsargmax, 666),
                                      output_tsargmax)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [0, 0], [0, 0], [0, 0], [0, 0]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)  # fillna
        result = (expected == output_tsargmax).all().item()
        self.assertTrue(result, 'test_argmax_ties failed')

    def test_argmin_change(self):
        output_tsargmax = tsf.ts_argmin(self.test_2d_change, 2)
        output_tsargmax = torch.where(torch.isnan(output_tsargmax), torch.full_like(output_tsargmax, 666),
                                      output_tsargmax)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [0, 1], [0, 1], [0, 1], [0, 1]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)  # fillna
        result = (expected == output_tsargmax).all().item()
        self.assertTrue(result, 'test_argmin_change failed')

    def test_argmin_ties(self):
        output_tsargmax = tsf.ts_argmin(self.test_2d_ties, 2)
        output_tsargmax = torch.where(torch.isnan(output_tsargmax), torch.full_like(output_tsargmax, 666),
                                      output_tsargmax)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [0, 0], [0, 0], [0, 0], [0, 0]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)  # fillna
        result = (expected == output_tsargmax).all().item()
        self.assertTrue(result, 'test_argmin_ties failed')

    def test_diff_normal(self):
        output_diff = tsf.diff(self.test_2d_change, 1)
        output_diff = torch.where(torch.isnan(output_diff), torch.full_like(output_diff, 666),
                                  output_diff)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [1, -1], [1, -1], [1, -1], [1, -1]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)
        result = (expected == output_diff).all().item()
        self.assertTrue(result, 'test_diff_normal failed')

    def test_diff_inverse(self):
        output_diff = tsf.diff(self.test_2d_change, -1)
        output_diff = torch.where(torch.isnan(output_diff), torch.full_like(output_diff, 666),
                                  output_diff)  # fillna
        expected = torch.tensor([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [np.nan, np.nan]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)
        result = (expected == output_diff).all().item()
        self.assertTrue(result, 'test_diff_inverse failed')

    def test_shift_forward(self):
        output_diff = tsf.shift(self.test_2d_change, 1)
        output_diff = torch.where(torch.isnan(output_diff), torch.full_like(output_diff, 666),
                                  output_diff)  # fillna
        expected = torch.tensor([[np.nan, np.nan], [1, 5], [2, 4], [3, 3], [4, 2]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)
        result = (expected == output_diff).all().item()
        self.assertTrue(result, 'test_shift_forward failed')

    def test_shift_backward(self):
        output_diff = tsf.shift(self.test_2d_change, -1)
        output_diff = torch.where(torch.isnan(output_diff), torch.full_like(output_diff, 666),
                                  output_diff)  # fillna
        expected = torch.tensor([[2, 4], [3, 3], [4, 2], [5, 1], [np.nan, np.nan]])
        expected = torch.where(torch.isnan(expected), torch.full_like(expected, 666), expected)
        result = (expected == output_diff).all().item()
        self.assertTrue(result, 'test_shift_backward failed')


if __name__ == '__main__':
    unittest.main()
