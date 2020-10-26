import torch
from featurizer.functions import time_series_functions as tsf
from featurizer.interface import Functor
import featurizer.functions.volume_price as vpf


class Alpha001(Functor):

    # (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    def forward(self, volume_ts, close_ts, open_ts):
        factor1 = tsf.rank(tsf.diff(torch.log(volume_ts), 1))
        factor2 = tsf.rank((close_ts - open_ts) / open_ts)
        output_tensor = tsf.rolling_corr(factor1, factor2, 6)
        return output_tensor


class Alpha002(Functor):

    # (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def forward(self, close_ts, high_ts, low_ts):
        output_tensor = (-1 * tsf.diff((((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts)), 1))
        return output_tensor


class Alpha003(Functor):

    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    def forward(self, close_ts, low_ts, high_ts):
        cond1 = close_ts == tsf.shift(close_ts, 1)
        cond2 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros(close_ts.size())
        consequance1 = close_ts - torch.min(low_ts, tsf.shift(close_ts, 1))
        consequance2 = close_ts - torch.max(high_ts, tsf.shift(close_ts, 1))
        inner = torch.where(cond2, consequance1, consequance2)
        output_tensor = torch.where(cond1, zeros, inner)
        return output_tensor


class Alpha004(Functor):

    def forward(self, close_ts, volume_ts):
        cond1 = (tsf.rolling_mean_(close_ts, 8) + tsf.rolling_std(close_ts, 8)) < tsf.rolling_mean_(close_ts, 2)
        cond2 = tsf.rolling_mean_(close_ts, 2) < (tsf.rolling_mean_(close_ts, 8) - tsf.rolling_std(close_ts, 8))
        cond3 = (volume_ts / tsf.rolling_mean_(volume_ts, 20)) >= 1
        ones = torch.ones(close_ts.size())
        output_tensor = torch.where([(cond1 | ~(cond1 & cond2 & cond3))], -1 * ones, ones)
        return output_tensor


class Alpha005(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 5), tsf.ts_rank(high_ts, 5), 5), 3)
        return output_tensor


class Alpha006(Functor):

    def forward(self, high_ts, open_ts):
        output_tensor = -1 * tsf.rank(torch.sign(tsf.diff((((open_ts * 0.85) + (high_ts * 0.15))), 4)))
        return output_tensor


# class Alpha007(Functor): # what max?
#
#     def forward(self, vmap_ts, close_ts, volume_ts):
#         output_tensor = ((tsf.rank(tsf.rolling_max((vmap_ts - close_ts), 3)) + tsf.rank(
#             ((vmap_ts - close_ts), 3))) * tsf.rank(tsf.diff(volume_ts, 3)))
#         return output_tensor


class Alpha008(Functor):

    def forward(self, vmap_ts, high_ts, low_ts):
        output_tensor = tsf.rank(tsf.diff(((((high_ts + low_ts) / 2) * 0.2) + (vmap_ts * 0.8)), 4) * -1)
        return output_tensor


class Alpha009(Functor):

    def forward(self, high_ts, low_ts, volume_ts):
        output_tensor = tsf.rolling_mean_(
            ((high_ts + low_ts) / 2 - (tsf.shift(high_ts, 1) + tsf.shif(low_ts, 1)) / 2) * (
                    high_ts - low_ts) / volume_ts, 7, 2)
        return output_tensor


# class Alpha010(Functor):  #  BUG in formula?
#
#     def forward(self, high_ts,return_ts,close_ts):
#         cond1 = return_ts<0
#         inner = torch.where(cond1,tsf.rolling_std(return_ts,20),close_ts)**2
#         output_tensor = tsf.rank(tsf.rolling_max(inner,5))
#         return output_tensor


class Alpha011(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         6)
        return output_tensor


class Alpha012(Functor):

    def forward(self, open_ts, close_ts, vmap_ts, **kwargs):
        output_tensor = (tsf.rank((open_ts - (tsf.rolling_sum_(vmap_ts, 10) / 10)))) * (
                    -1 * (tsf.rank(abs((close_ts - vmap_ts)))))
        return output_tensor


class Alpha013(Functor):

    def forward(self, high_ts, low_ts, vmap_ts):
        output_tensor = (((high_ts * low_ts) ^ 0.5) - vmap_ts)
        return output_tensor


class Alpha014(Functor):

    def forward(self, close_ts):
        output_tensor = close_ts - tsf.shift(close_ts, 5)
        return output_tensor


class Alpha015(Functor):

    def forward(self, close_ts, open_ts):
        output_tensor = open_ts / tsf.shift(close_ts, 1) - 1
        return output_tensor


class Alpha016(Functor):

    def forward(self, vwap_ts, volume_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rank(tsf.rolling_corr(tsf.rank(volume_ts), tsf.rank(vwap_ts), 5)), 5)
        return output_tensor


# class Alpha017(Functor): # what max?
#
#     def forward(self,vwap_ts,close_ts):
#         output_tensor = tsf.rank((vwap_ts - tsf.rolling_max(vwap_ts, 15)))^tsf.diff(close_ts, 5)
#         return output_tensor


class Alpha018(Functor):

    def forward(self, close_ts):
        output_tensor = close_ts / tsf.shift(close_ts, 5)
        return output_tensor


class Alpha019(Functor):

    def forward(self, close_ts):
        cond1 = close_ts < tsf.shift(close_ts, 5)
        cond2 = close_ts == tsf.shift(close_ts, 5)
        inner = (close_ts - tsf.shift(close_ts, 5)) / tsf.shift(close_ts, 5)
        zeros = torch.zeros(close_ts.size())
        output_tensor = torch.where(cond1, inner, (close_ts - tsf.shift(close_ts, 5)) / close_ts)
        output_tensor = torch.where([~cond1 & cond2], zeros, output_tensor)
        return output_tensor


class Alpha020(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.shift(close_ts, 6)) / tsf.shift(close_ts, 6) * 100
        return output_tensor


# class Alpha021(Functor):  # formula BUG
#
#     def forward(self, close_ts):
#         output_tensor = REGBETA(tsf.rolling_mean_(close_ts,6),SEQUENCE(6))
#         return output_tensor

