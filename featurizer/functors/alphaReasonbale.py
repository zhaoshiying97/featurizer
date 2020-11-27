import torch

from featurizer.functions import time_series_functions as tsf
from featurizer.interface import Functor


class kd_PVDeviation_change(Functor):
    """
    The level of price-volume-change deviation in the recent 'ROLL_PERIOD' days.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, volume_ts, return_ts):
        output_tensor = (-1 * tsf.rolling_corr(tsf.diff(torch.log(volume_ts), 1), return_ts), self.roll_period)
        return output_tensor


class kd_PVDeviation_value(Functor):
    """
    The level of price-volume-value deviation in the recent 'ROLL_PERIOD' days.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, volume_ts, close_ts):
        output_tensor = (-1 * tsf.rolling_corr(torch.log(volume_ts), close_ts), self.roll_period)
        return output_tensor


class kd_LongShortStrength_plain(Functor):
    """
    The indicator of the relative strength of long-short counterparties.
    """

    def forward(self, close_ts, high_ts, low_ts):
        output_tensor = (-1 * tsf.diff((((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts)), 1))
        return output_tensor


class kd_PVDeviation_rankTsMax(Functor):
    """
    The maximum correlation between the RANK_PERIOD-day ts_rank of volume and high price in ROLL_PERIOD days.
    """

    def __init__(self, rank_period, roll_period):
        self.roll_peridod = roll_period
        self.rank_period = rank_period

    def forward(self, volume_ts, high_ts, rank_period):
        output_tensor = -1 * tsf.rolling_max(
            tsf.rolling_corr(tsf.ts_rank(volume_ts, rank_period), tsf.ts_rank(high_ts, rank_period), self.roll_period),
            self.roll_period)
        return output_tensor


class kd_meanReversion_isCloseDeviate(Functor):
    """
    Whether the close price deviate significantly from its ROLL_PERIOD-day mean.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts):
        cond1 = (tsf.rolling_mean_(close_ts, self.roll_period) + tsf.rolling_std(close_ts, self.roll_period)) < close_ts
        cond2 = close_ts < (tsf.rolling_mean_(close_ts, self.roll_period) - tsf.rolling_std(close_ts, self.roll_period))
        ones = torch.ones(close_ts.size())
        zeros = torch.zeros(close_ts.size())
        output_tensor = torch.where((cond1 & cond2), ones, zeros)
        return output_tensor


class kd_PVTheory_isVolumeHigh(Functor):
    """
    Whether the volume is higher than the ROLL_PERIOD-day average
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, volume_ts):
        cond1 = (volume_ts / tsf.rolling_mean_(volume_ts, self.roll_period)) >= 1
        ones = torch.ones(volume_ts.size())
        output_tensor = torch.where(cond1, ones, -1 * ones)
        return output_tensor


class kd_longShortStrength_volumeSum(Functor):
    """
    The ROLL_PERIOD-day summation of the volume-weighted long-short relative strength
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         self.roll_period)
        return output_tensor


class kd_PVTheory_geoMinusVolumeWeighted(Functor):
    """
    The difference between the geometric average price and the volume weighted average price
    """

    def forward(self, high_ts, low_ts, vwap_ts):
        output_tensor = (high_ts * low_ts) ** 0.5 - vwap_ts
        return output_tensor


class kd_momentum_closeValue(Functor):
    """
    The ROLL_PERIOD-day close price difference.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts):
        output_tensor = tsf.diff(close_ts, self.roll_period)
        return output_tensor


class kd_momentum_openSurprise(Functor):
    """
    The open surprise.
    """

    def forward(self, close_ts, open_ts):
        output_tensor = open_ts / tsf.shift(close_ts, 1) - 1
        return output_tensor


class kd_momentum_closePercentage(Functor):
    """
    The ROLL_PERIOD-day close price difference, in percentage.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts):
        output_tensor = close_ts / tsf.shift(close_ts, self.roll_period)
        return output_tensor


class kd_momentum_diffFromMean(Functor):
    """
    The difference between the close price and its ROLL_PERIOD-day mean.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, self.roll_period)) / tsf.rolling_mean(close_ts,
                                                                                                      self.roll_period) * 100
        return output_tensor


class kd_reversion_abnormalhigh(Functor):
    """
    If the HIGH is larger than its ROLL_PERIOD-day mean(abnormal high),return its two-day increase.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, high_ts):
        cond = tsf.rolling_mean(high_ts, self.roll_period) < high_ts
        zeros = torch.zeros(high_ts.size())
        output_tensor = torch.where(cond, (-1 * tsf.diff(high_ts, 2)), zeros)
        return output_tensor


class kd_PVTheory_volumeRatio(Functor):
    """
    The ratio of the total volume when price is rising to the volume when price is decreasing in ROLL_PERIOD days,
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts, volume_ts):
        cond1 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros(close_ts.size())
        inner1 = torch.where(cond1, volume_ts, zeros)
        cond2 = close_ts <= tsf.shift(close_ts, 1)
        inner2 = tsf.rolling_sum_(torch.where(cond2, volume_ts, zeros), self.roll_period)
        output_tensor = inner1 / inner2
        return output_tensor


class kd_PVTheory_volumeDiff(Functor):
    """
    The ROLL_PREIOD-day summation of the difference between the volume when price is rising and the volume when price is decreasing.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts, volume_ts):
        sign = torch.sign(tsf.diff(close_ts, 1))
        output_tensor = tsf.rolling_sum_(sign * volume_ts, self.roll_period)
        return output_tensor


class kd_std_volume(Functor):
    """
    The ROLL_PERIOD-day standard error of volume.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, volume_ts):
        output_tensor = tsf.rolling_std(volume_ts, self.roll_period)
        return output_tensor


class kd_reversion_distanceToLowest(Functor):
    """
    The distance to the period-day lowest price, in percentage.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, low_ts):
        output_tensor = (tsf.ts_argmin(low_ts, self.roll_period) / self.roll_period) * 100
        return output_tensor


class kd_longShortStrength_sumShadowRatio(Functor):
    """
    The roll_period-day summation of the ratio of the upper shadow to lower shadow.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, high_ts, low_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        output_tensor = tsf.rolling_sum_(torch.max(zeros, high_ts - tsf.shift(close_ts, 1)),
                                         self.roll_period) / tsf.rolling_sum_(
            torch.max(zeros, tsf.shift(close_ts, 1) - low_ts), self.roll_period) * 100
        return output_tensor


class kd_momentum_avgPrice(Functor):
    """
    (CLOSE+HIGH+LOW)/3
    """

    def forward(self, high_ts, low_ts, close_ts):
        output_tensor = (close_ts + high_ts + low_ts) / 3
        return output_tensor


class kd_reversion_totalDecrease(Functor):
    """
    The ROLL_PERIOD-day summation of the price decrease.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, high_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        cond = tsf.diff(close_ts, 1) < 0
        output_tensor = tsf.rolling_sum_(torch.where(cond, torch.abs(tsf.diff(close_ts, 1)), zeros), self.roll_period)
        return output_tensor


class kd_reversion_totalIncrease(Functor):
    """
    The ROLL_PERIOD-day summation of the price increase.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts):
        cond = tsf.diff(close_ts, 1) > 0
        zeros = torch.zeros(close_ts.size())
        output_tensor = tsf.rolling_sum_(torch.where(cond, close_ts - tsf.shift(close_ts, 1), zeros), self.roll_period)
        return output_tensor


class kd_reversion_distanceToHighest(Functor):
    """
    The distance to the period-day highest price, in percentage.
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, high_ts):
        output_tensor = (tsf.ts_argmax(high_ts, self.roll_period) / self.roll_period) * 100
        return output_tensor


class kd_reversion_avgDiffFromMean(Functor):
    """
    The ROLL_PERIOD-day average difference of close to its mean
    """

    def __init__(self, roll_period):
        self.roll_period = roll_period

    def forward(self, close_ts):
        output_tensor = tsf.rolling_mean_(torch.abs(close_ts - tsf.rolling_mean_(close_ts, self.roll_period)),
                                          self.roll_period)
        return output_tensor
