import torch

from featurizer.functions import time_series_functions as tsf
from featurizer.interface import Functor


class kd_PVDeviation_change_shortterm(Functor):

    def forward(self, volume_ts, return_ts):
        output_tensor = (-1 * tsf.rolling_corr(tsf.diff(torch.log(volume_ts), 1), return_ts), 5)
        return output_tensor


class kd_PVDeviation_change_midterm(Functor):

    def forward(self, volume_ts, return_ts):
        output_tensor = (-1 * tsf.rolling_corr(tsf.diff(torch.log(volume_ts), 1), return_ts), 10)
        return output_tensor


class kd_PVDeviation_change_longterm(Functor):

    def forward(self, volume_ts, return_ts):
        output_tensor = (-1 * tsf.rolling_corr(tsf.diff(torch.log(volume_ts), 1), return_ts), 20)
        return output_tensor


class kd_PVDeviation_change_cterm(Functor):

    def forward(self, volume_ts, return_ts, roll_period):
        output_tensor = (-1 * tsf.rolling_corr(tsf.diff(torch.log(volume_ts), 1), return_ts), roll_period)
        return output_tensor


class kd_PVDeviation_value_shortterm(Functor):

    def forward(self, volume_ts, close_ts):
        output_tensor = (-1 * tsf.rolling_corr(torch.log(volume_ts), close_ts), 5)
        return output_tensor


class kd_PVDeviation_value_midterm(Functor):

    def forward(self, volume_ts, close_ts):
        output_tensor = (-1 * tsf.rolling_corr(torch.log(volume_ts), close_ts), 10)
        return output_tensor


class kd_PVDeviation_value_longterm(Functor):

    def forward(self, volume_ts, close_ts):
        output_tensor = (-1 * tsf.rolling_corr(torch.log(volume_ts), close_ts), 20)
        return output_tensor


class kd_PVDeviation_value_cterm(Functor):

    def forward(self, volume_ts, close_ts, roll_period):
        output_tensor = (-1 * tsf.rolling_corr(torch.log(volume_ts), close_ts), roll_period)
        return output_tensor


class kd_LongShortStrength_plain(Functor):

    def forward(self, close_ts, high_ts, low_ts):
        output_tensor = (-1 * tsf.diff((((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts)), 1))
        return output_tensor


class kd_PVDeviation_rankTsMax_shortrank_shortterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 5), tsf.ts_rank(high_ts, 5), 5), 3)
        return output_tensor


class kd_PVDeviation_rankTsMax_midrank_shortterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 10), tsf.ts_rank(high_ts, 10), 5),
                                             3)
        return output_tensor


class kd_PVDeviation_rankTsMax_longrank_shortterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 20), tsf.ts_rank(high_ts, 20), 5),
                                             3)
        return output_tensor


class kd_PVDeviation_rankTsMax_shortrank_midterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 5), tsf.ts_rank(high_ts, 5), 10),
                                             10)
        return output_tensor


class kd_PVDeviation_rankTsMax_midrank_midterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 10), tsf.ts_rank(high_ts, 10), 10), 10)
        return output_tensor


class kd_PVDeviation_rankTsMax_longrank_midterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 20), tsf.ts_rank(high_ts, 20), 10),
                                             10)
        return output_tensor


class kd_PVDeviation_rankTsMax_shortrank_longterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 5), tsf.ts_rank(high_ts, 5), 20),
                                             20)
        return output_tensor


class kd_PVDeviation_rankTsMax_midrank_longterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 10), tsf.ts_rank(high_ts, 10), 20),
                                             20)
        return output_tensor


class kd_PVDeviation_rankTsMax_longrank_longterm(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 20), tsf.ts_rank(high_ts, 20), 20),
                                             20)
        return output_tensor


class kd_PVDeviation_rankTsMax_crank_cterm(Functor):

    def forward(self, volume_ts, high_ts, rank_period, roll_period):
        output_tensor = -1 * tsf.rolling_max(
            tsf.rolling_corr(tsf.ts_rank(volume_ts, rank_period), tsf.ts_rank(high_ts, rank_period), roll_period),
            roll_period)
        return output_tensor


class kd_MeanReversion_IsCloseDeviate_shortterm(Functor):

    def forward(self, close_ts):
        cond1 = (tsf.rolling_mean_(close_ts, 5) + tsf.rolling_std(close_ts, 5)) < close_ts
        cond2 = close_ts < (tsf.rolling_mean_(close_ts, 5) - tsf.rolling_std(close_ts, 5))
        ones = torch.ones(close_ts.size())
        zeros = torch.zeros(close_ts.size())
        output_tensor = torch.where((cond1 & cond2), ones, zeros)
        return output_tensor


class kd_MeanReversion_IsCloseDeviate_midterm(Functor):

    def forward(self, close_ts):
        cond1 = (tsf.rolling_mean_(close_ts, 10) + tsf.rolling_std(close_ts, 10)) < close_ts
        cond2 = close_ts < (tsf.rolling_mean_(close_ts, 10) - tsf.rolling_std(close_ts, 10))
        ones = torch.ones(close_ts.size())
        zeros = torch.zeros(close_ts.size())
        output_tensor = torch.where((cond1 & cond2), ones, zeros)
        return output_tensor


class kd_MeanReversion_IsCloseDeviate_longterm(Functor):

    def forward(self, close_ts):
        cond1 = (tsf.rolling_mean_(close_ts, 20) + tsf.rolling_std(close_ts, 20)) < close_ts
        cond2 = close_ts < (tsf.rolling_mean_(close_ts, 20) - tsf.rolling_std(close_ts, 20))
        ones = torch.ones(close_ts.size())
        zeros = torch.zeros(close_ts.size())
        output_tensor = torch.where((cond1 & cond2), ones, zeros)
        return output_tensor


class kd_MeanReversion_IsCloseDeviate_cterm(Functor):

    def forward(self, close_ts, roll_period):
        cond1 = (tsf.rolling_mean_(close_ts, roll_period) + tsf.rolling_std(close_ts, roll_period)) < close_ts
        cond2 = close_ts < (tsf.rolling_mean_(close_ts, roll_period) - tsf.rolling_std(close_ts, roll_period))
        ones = torch.ones(close_ts.size())
        zeros = torch.zeros(close_ts.size())
        output_tensor = torch.where((cond1 & cond2), ones, zeros)
        return output_tensor


class kd_PVTheory_IsVolumeHigh_shortterm(Functor):

    def forward(self, volume_ts):
        cond1 = (volume_ts / tsf.rolling_mean_(volume_ts, 5)) >= 1
        ones = torch.ones(volume_ts.size())
        output_tensor = torch.where(cond1, ones, -1 * ones)
        return output_tensor


class kd_PVTheory_IsVolumeHigh_midterm(Functor):

    def forward(self, volume_ts):
        cond1 = (volume_ts / tsf.rolling_mean_(volume_ts, 10)) >= 1
        ones = torch.ones(volume_ts.size())
        output_tensor = torch.where(cond1, ones, -1 * ones)
        return output_tensor


class kd_PVTheory_IsVolumeHigh_longterm(Functor):

    def forward(self, volume_ts):
        cond1 = (volume_ts / tsf.rolling_mean_(volume_ts, 20)) >= 1
        ones = torch.ones(volume_ts.size())
        output_tensor = torch.where(cond1, ones, -1 * ones)
        return output_tensor


class kd_PVTheory_IsVolumeHigh_cterm(Functor):

    def forward(self, volume_ts, roll_period):
        cond1 = (volume_ts / tsf.rolling_mean_(volume_ts, roll_period)) >= 1
        ones = torch.ones(volume_ts.size())
        output_tensor = torch.where(cond1, ones, -1 * ones)
        return output_tensor


class kd_longShortStrength_volumeSum_shortterm(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         5)
        return output_tensor


class kd_longShortStrength_volumeSum_midterm(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         10)
        return output_tensor


class kd_longShortStrength_volumeSum_longterm(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         20)
        return output_tensor


class kd_longShortStrength_volumeSum_cterm(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts, roll_period):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         roll_period)
        return output_tensor


class kd_PVTheory_geoMinusVolumeWeighted(Functor):

    def forward(self, high_ts, low_ts, vwap_ts):
        output_tensor = (high_ts * low_ts) ** 0.5 - vwap_ts
        return output_tensor


class kd_momentum_closeValue_shortterm(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.diff(close_ts, 5)
        return output_tensor


class kd_momentum_closeValue_midterm(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.diff(close_ts, 10)
        return output_tensor


class kd_momentum_closeValue_longterm(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.diff(close_ts, 20)
        return output_tensor


class kd_momentum_closeValue_cterm(Functor):

    def forward(self, close_ts, roll_period):
        output_tensor = tsf.diff(close_ts, roll_period)
        return output_tensor


class kd_momentum_openSurprise(Functor):

    def forward(self, close_ts, open_ts):
        output_tensor = open_ts / tsf.shift(close_ts, 1) - 1
        return output_tensor


class kd_momentum_closePercentage_shortterm(Functor):

    def forward(self, close_ts):
        output_tensor = close_ts / tsf.shift(close_ts, 5)
        return output_tensor


class kd_momentum_closePercentage_midterm(Functor):

    def forward(self, close_ts):
        output_tensor = close_ts / tsf.shift(close_ts, 10)
        return output_tensor


class kd_momentum_closePercentage_longterm(Functor):

    def forward(self, close_ts):
        output_tensor = close_ts / tsf.shift(close_ts, 20)
        return output_tensor


class kd_momentum_closePercentage_cterm(Functor):

    def forward(self, close_ts, roll_period):
        output_tensor = close_ts / tsf.shift(close_ts, roll_period)
        return output_tensor


class kd_momentum_diffFromMean_shortterm(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, 5)) / tsf.rolling_mean(close_ts, 5) * 100
        return output_tensor


class kd_momentum_diffFromMeange_midterm(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, 10)) / tsf.rolling_mean(close_ts, 10) * 100
        return output_tensor


class kd_momentum_diffFromMeange_longterm(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, 20)) / tsf.rolling_mean(close_ts, 20) * 100
        return output_tensor


class kd_momentum_diffFroMean_cterm(Functor):

    def forward(self, close_ts, roll_period):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, roll_period)) / tsf.rolling_mean(close_ts,
                                                                                                 roll_period) * 100
        return output_tensor


class kd_reversion_abnormalhigh_shortterm(Functor):

    def forward(self, high_ts):
        cond = tsf.rolling_mean(high_ts, 5) < high_ts
        zeros = torch.zeros(high_ts.size())
        output_tensor = torch.where(cond, (-1 * tsf.diff(high_ts, 2)), zeros)
        return output_tensor


class kd_reversion_abnormalhigh_midterm(Functor):

    def forward(self, high_ts):
        cond = tsf.rolling_mean(high_ts, 10) < high_ts
        zeros = torch.zeros(high_ts.size())
        output_tensor = torch.where(cond, (-1 * tsf.diff(high_ts, 2)), zeros)
        return output_tensor


class kd_reversion_abnormalhigh_longterm(Functor):

    def forward(self, high_ts):
        cond = tsf.rolling_mean(high_ts, 20) < high_ts
        zeros = torch.zeros(high_ts.size())
        output_tensor = torch.where(cond, (-1 * tsf.diff(high_ts, 2)), zeros)
        return output_tensor


class kd_reversion_abnormalhigh_cterm(Functor):

    def forward(self, high_ts, roll_period):
        cond = tsf.rolling_mean(high_ts, roll_period) < high_ts
        zeros = torch.zeros(high_ts.size())
        output_tensor = torch.where(cond, (-1 * tsf.diff(high_ts, 2)), zeros)
        return output_tensor


class kd_PVTheory_volumeRatio_shortterm(Functor):

    def forward(self, close_ts, volume_ts):
        cond1 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros(close_ts.size())
        inner1 = torch.where(cond1, volume_ts, zeros)
        cond2 = close_ts <= tsf.shift(close_ts, 1)
        inner2 = tsf.rolling_sum_(torch.where(cond2, volume_ts, zeros), 5)
        output_tensor = inner1 / inner2
        return output_tensor


class kd_PVTheory_volumeRatio_midterm(Functor):

    def forward(self, close_ts, volume_ts):
        cond1 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros(close_ts.size())
        inner1 = torch.where(cond1, volume_ts, zeros)
        cond2 = close_ts <= tsf.shift(close_ts, 1)
        inner2 = tsf.rolling_sum_(torch.where(cond2, volume_ts, zeros), 10)
        output_tensor = inner1 / inner2
        return output_tensor


class kd_PVTheory_volumeRatio_longterm(Functor):

    def forward(self, close_ts, volume_ts):
        cond1 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros(close_ts.size())
        inner1 = torch.where(cond1, volume_ts, zeros)
        cond2 = close_ts <= tsf.shift(close_ts, 1)
        inner2 = tsf.rolling_sum_(torch.where(cond2, volume_ts, zeros), 20)
        output_tensor = inner1 / inner2
        return output_tensor


class kd_PVTheory_volumeRatio_cterm(Functor):

    def forward(self, close_ts, volume_ts, roll_period):
        cond1 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros(close_ts.size())
        inner1 = torch.where(cond1, volume_ts, zeros)
        cond2 = close_ts <= tsf.shift(close_ts, 1)
        inner2 = tsf.rolling_sum_(torch.where(cond2, volume_ts, zeros), roll_period)
        output_tensor = inner1 / inner2
        return output_tensor


class kd_PVTheory_volumeDiff_shortterm(Functor):

    def forward(self, close_ts, volume_ts):
        sign = torch.sign(tsf.diff(close_ts, 1))
        output_tensor = tsf.rolling_sum_(sign * volume_ts, 5)
        return output_tensor


class kd_PVTheory_volumeDiff_midterm(Functor):

    def forward(self, close_ts, volume_ts):
        sign = torch.sign(tsf.diff(close_ts, 1))
        output_tensor = tsf.rolling_sum_(sign * volume_ts, 10)
        return output_tensor


class kd_PVTheory_volumeDiff_longterm(Functor):

    def forward(self, close_ts, volume_ts):
        sign = torch.sign(tsf.diff(close_ts, 1))
        output_tensor = tsf.rolling_sum_(sign * volume_ts, 20)
        return output_tensor


class kd_PVTheory_volumeDiff_cterm(Functor):

    def forward(self, close_ts, volume_ts, roll_period):
        sign = torch.sign(tsf.diff(close_ts, 1))
        output_tensor = tsf.rolling_sum_(sign * volume_ts, roll_period)
        return output_tensor


class kd_std_volume_shortterm(Functor):

    def forward(self, volume_ts):
        output_tensor = tsf.rolling_std(volume_ts, 5)
        return output_tensor


class kd_std_volume_midterm(Functor):

    def forward(self, volume_ts):
        output_tensor = tsf.rolling_std(volume_ts, 10)
        return output_tensor


class kd_std_volume_longterm(Functor):

    def forward(self, volume_ts):
        output_tensor = tsf.rolling_std(volume_ts, 20)
        return output_tensor


class kd_std_volume_cterm(Functor):

    def forward(self, volume_ts,roll_period):
        output_tensor = tsf.rolling_std(volume_ts, roll_period)
        return output_tensor


class kd_reversion_distanceToLowest_shortterm(Functor):

    def forward(self, low_ts):
        output_tensor = (tsf.ts_argmin(low_ts, 5) / 5) * 100
        return output_tensor


class kd_reversion_distanceToLowest_midterm(Functor):

    def forward(self, low_ts):
        output_tensor = (tsf.ts_argmin(low_ts, 10) / 10) * 100
        return output_tensor


class kd_reversion_distanceToLowest_longterm(Functor):

    def forward(self, low_ts):
        output_tensor = (tsf.ts_argmin(low_ts, 20) / 20) * 100
        return output_tensor


class kd_reversion_distanceToLowest_cterm(Functor):

    def forward(self, low_ts,roll_period):
        output_tensor = (tsf.ts_argmin(low_ts, roll_period) / roll_period) * 100
        return output_tensor


class kd_LongShortStrength_sumShadowRatio_shortterm(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        output_tensor = tsf.rolling_sum_(torch.max(zeros, high_ts - tsf.shift(close_ts, 1)), 5) / tsf.rolling_sum_(
            torch.max(zeros, tsf.shift(close_ts, 1) - low_ts), 5) * 100
        return output_tensor


class kd_LongShortStrength_sumShadowRatio_midterm(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        output_tensor = tsf.rolling_sum_(torch.max(zeros, high_ts - tsf.shift(close_ts, 1)), 10) / tsf.rolling_sum_(
            torch.max(zeros, tsf.shift(close_ts, 1) - low_ts), 10) * 100
        return output_tensor


class kd_LongShortStrength_sumShadowRatio_longterm(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        output_tensor = tsf.rolling_sum_(torch.max(zeros, high_ts - tsf.shift(close_ts, 1)), 20) / tsf.rolling_sum_(
            torch.max(zeros, tsf.shift(close_ts, 1) - low_ts), 20) * 100
        return output_tensor


class kd_LongShortStrength_sumShadowRatio_cterm(Functor):

    def forward(self, high_ts, low_ts, close_ts,roll_period):
        zeros = torch.zeros(high_ts.size())
        output_tensor = tsf.rolling_sum_(torch.max(zeros, high_ts - tsf.shift(close_ts, 1)), roll_period) / tsf.rolling_sum_(
            torch.max(zeros, tsf.shift(close_ts, 1) - low_ts), roll_period) * 100
        return output_tensor


class kd_momentum_avgPrice(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        output_tensor = (close_ts + high_ts + low_ts) / 3
        return output_tensor


class kd_reversion_totalDecrease_shortterm(Functor):

    def forward(self, high_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        cond = tsf.diff(close_ts, 1) < 0
        output_tensor = tsf.rolling_sum_(torch.where(cond, torch.abs(tsf.diff(close_ts, 1)), zeros), 5)
        return output_tensor


class kd_reversion_totalDecrease_midterm(Functor):

    def forward(self, high_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        cond = tsf.diff(close_ts, 1) < 0
        output_tensor = tsf.rolling_sum_(torch.where(cond, torch.abs(tsf.diff(close_ts, 1)), zeros), 10)
        return output_tensor


class kd_reversion_totalDecrease_longterm(Functor):

    def forward(self, high_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        cond = tsf.diff(close_ts, 1) < 0
        output_tensor = tsf.rolling_sum_(torch.where(cond, torch.abs(tsf.diff(close_ts, 1)), zeros), 20)
        return output_tensor


class kd_reversion_totalDecrease_cterm(Functor):

    def forward(self, high_ts, close_ts,roll_period):
        zeros = torch.zeros(high_ts.size())
        cond = tsf.diff(close_ts, 1) < 0
        output_tensor = tsf.rolling_sum_(torch.where(cond, torch.abs(tsf.diff(close_ts, 1)), zeros), roll_period)
        return output_tensor


class kd_reversion_totalIncrease_shortterm(Functor):

    def forward(self, close_ts):
        cond = tsf.diff(close_ts, 1) > 0
        zeros = torch.zeros(close_ts.size())
        output_tensor = tsf.rolling_sum_(torch.where(cond, close_ts - tsf.shift(close_ts, 1), zeros), 5)
        return output_tensor


class kd_reversion_totalIncrease_midterm(Functor):

    def forward(self, close_ts):
        cond = tsf.diff(close_ts, 1) > 0
        zeros = torch.zeros(close_ts.size())
        output_tensor = tsf.rolling_sum_(torch.where(cond, close_ts - tsf.shift(close_ts, 1), zeros), 10)
        return output_tensor


class kd_reversion_totalIncrease_longterm(Functor):

    def forward(self, close_ts):
        cond = tsf.diff(close_ts, 1) > 0
        zeros = torch.zeros(close_ts.size())
        output_tensor = tsf.rolling_sum_(torch.where(cond, close_ts - tsf.shift(close_ts, 1), zeros), 20)
        return output_tensor


class kd_reversion_totalIncrease_cterm(Functor):

    def forward(self, close_ts,roll_period):
        cond = tsf.diff(close_ts, 1) > 0
        zeros = torch.zeros(close_ts.size())
        output_tensor = tsf.rolling_sum_(torch.where(cond, close_ts - tsf.shift(close_ts, 1), zeros), roll_period)
        return output_tensor


class kd_reversion_distanceToHighest_shortterm(Functor):

    def forward(self, high_ts):
        output_tensor = (tsf.ts_argmax(high_ts, 5) / 5) * 100
        return output_tensor


class kd_reversion_distanceToHighest_midterm(Functor):

    def forward(self, high_ts):
        output_tensor = (tsf.ts_argmax(high_ts, 10) / 10) * 100
        return output_tensor


class kd_reversion_distanceToHighest_longterm(Functor):

    def forward(self, high_ts):
        output_tensor = (tsf.ts_argmax(high_ts, 20) / 20) * 100
        return output_tensor


class kd_reversion_distanceToHighest_cterm(Functor):

    def forward(self, high_ts,roll_period):
        output_tensor = (tsf.ts_argmax(high_ts, roll_period) / roll_period) * 100
        return output_tensor


class kd_reversion_avgDifffrommean_shortterm(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.rolling_mean_(torch.abs(close_ts - tsf.rolling_mean_(close_ts, 5)), 5)
        return output_tensor


class kd_reversion_avgDifffrommean_midterm(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.rolling_mean_(torch.abs(close_ts - tsf.rolling_mean_(close_ts, 10)), 10)
        return output_tensor


class kd_reversion_avgDifffrommean_longterm(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.rolling_mean_(torch.abs(close_ts - tsf.rolling_mean_(close_ts, 20)), 20)
        return output_tensor


class kd_reversion_avgDifffrommean_cterm(Functor):

    def forward(self, close_ts,roll_period):
        output_tensor = tsf.rolling_mean_(torch.abs(close_ts - tsf.rolling_mean_(close_ts, roll_period)), roll_period)
        return output_tensor