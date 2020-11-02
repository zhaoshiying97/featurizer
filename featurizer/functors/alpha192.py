import torch

from featurizer.functions import time_series_functions as tsf
from featurizer.interface import Functor


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


# class Alpha025(Functor): # decay_linear
#
#     def forword(self):
#         ((-1 * tsf.rank((tsf.shift(CLOSE, 7) * (1 - tsf.rank(DECAYLINEAR((VOLUME / MEAN(VOLUME, 20)), 9)))))) * (
#                     1 + RANK(SUM(RET, 250))))

class Alpha026(Functor):

    def forword(self, close_ts, vwap_ts):
        output_tensor = ((((tsf.rolling_sum_(close_ts, 7) / 7) - close_ts)) + (
            (tsf.rolling_corr(vwap_ts, tsf.shift(close_ts, 5), 230))))
        return output_tensor


class Alpha029(Functor):

    def forword(self, close_ts, volume_ts):
        output_tensor = (close_ts - tsf.shift(close_ts, 6)) / tsf.shift(close_ts, 6) * volume_ts
        return output_tensor


class Alpha031(Functor):

    def forword(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, 12)) / tsf.rolling_mean(close_ts, 12) * 100
        return output_tensor


class Alpha032(Functor):

    def forword(self, high_ts, volume_ts):
        output_tensor = (
                -1 * tsf.rolling_sum_(tsf.rank(tsf.rolling_corr(tsf.rank(high_ts), tsf.rank(volume_ts), 3)), 3))
        return output_tensor


class Alpha033(Functor):

    def forword(self, low_ts, return_ts, volume_ts):
        output_tensor = ((((-1 * tsf.rolling_min(low_ts, 5)) + tsf.rank(tsf.rolling_min(low_ts, 5), 5)) * tsf.rank(
            ((tsf.rolling_sum_(return_ts, 240) - tsf.rolling_sum_(return_ts, 20)) / 220))) * tsf.rolling_min(volume_ts,
                                                                                                             5))
        return output_tensor


class Alpha034(Functor):
    def forword(self, close_ts):
        output_tensor = tsf.rolling_mean(close_ts, 12) / close_ts
        return output_tensor


class Alpha036(Functor):

    def forword(self, volume_ts, vwap_ts):
        output_tensor = tsf.rank(tsf.rolling_sum_(tsf.rolling_corr(tsf.rank(volume_ts), tsf.rank(vwap_ts)), 6), 2)
        return output_tensor


class Alpha037(Functor):

    def forword(self, open_ts, return_ts):
        output_tensor = (-1 * tsf.rank(((tsf.rolling_sum_(open_ts, 5) * tsf.rolling_sum_(return_ts, 5)) - tsf.shift(
            (tsf.rolling_sum_(open_ts, 5) * tsf.rolling_sum_(return_ts, 5)), 10))))
        return output_tensor


class Alpha038(Functor):

    def forword(self, high_ts):
        cond = tsf.rolling_mean(high_ts, 20) < high_ts
        zeros = torch.zeros(high_ts.size())
        output_tensor = torch.where(cond, (-1 * tsf.diff(high_ts, 2)), zeros)
        return output_tensor


class Alpha040(Functor):

    def forword(self, close_ts, volume_ts):
        cond1 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros(close_ts.size())
        inner1 = torch.where(cond1, volume_ts, zeros)
        cond2 = close_ts <= tsf.shift(close_ts, 1)
        inner2 = tsf.rolling_sum_(torch.where(cond2, volume_ts, zeros), 20)
        output_tensor = inner1 / inner2
        return output_tensor


class Alpha041(Functor):

    def forword(self, vwap_ts):
        output_tensor = (tsf.rank(tsf.rolling_max(tsf.diff((vwap_ts), 3), 5)) * -1)
        return output_tensor


class Alpha042(Functor):

    def forword(self, high_ts, volume_ts):
        output_tensor = ((-1 * tsf.rank(tsf.rolling_std(high_ts, 10))) * tsf.rolling_corr(high_ts, volume_ts, 10))
        return output_tensor


class Alpha043(Functor):

    def forword(self, close_ts, volume_ts):
        sign = torch.sign(tsf.diff(close_ts, 1))
        output_tensor = sign * volume_ts
        return output_tensor


class Alpha045(Functor):

    def forword(self, close_ts, open_ts, vwap_ts, volume_ts):
        output_tensor = (tsf.rank(tsf.diff((((close_ts * 0.6) + (open_ts * 0.4))), 1)) * tsf.rank(
            tsf.rolling_corr(vwap_ts, tsf.rolling_mean_(volume_ts, 150), 15)))
        return output_tensor


class Alpha046(Functor):

    def forword(self, close_ts):
        output_tensor = (tsf.rolling_mean_(close_ts, 3) + tsf.rolling_mean_(close_ts, 6) + tsf.rolling_mean_(close_ts,
                                                                                                             12) + tsf.rolling_mean_(
            close_ts, 24)) / (4 * close_ts)
        return output_tensor


class Alpha048(Functor):

    def forword(self, close_ts, volume_ts):
        output_tensor = (-1 * ((tsf.rank(((torch.sign((close_ts - tsf.shift(close_ts, 1))) + torch.sign(
            (tsf.shift(close_ts, 1) - tsf.shift(close_ts, 2)))) + torch.sign(
            (tsf.shift(close_ts, 2) - torch.shift(close_ts, 3)))))) * tsf.rolling_sum_(volume_ts,
                                                                                       5)) / tsf.rolling_sum_(volume_ts,
                                                                                                              20))
        return output_tensor


class Alpha049(Functor):

    def forword(self, high_ts, low_ts):
        output_tensor = torch.max(tsf.diff(high_ts, 1), tsf.diff(low_ts, 1))
        return output_tensor


# class Alpha050(Functor):  #formula bug
#
#     def forword(self,high_ts,low_ts):
#         zeros=torch.zeros(high_ts.size())
#         cond1=(high_ts+low_ts)<=tsf.shift(high_ts,1)+tsf.shift(low_ts,1)
#         inner1=tsf.rolling_sum_(torch.where(cond1,zeros,torch.max(torch.abs(high_ts-tsf.shift(high_ts,1)),torch.abs(low_ts-tsf.shift(low_ts,1)))),12)
#         inner2=tsf.rolling_sum_(torch.where(cond1,zeros,torch.max(torch.abs(high_ts-tsf.shift(high_ts,1)),torch.abs(low_ts-tsf.shift(low_ts,1)))),12)
#         inner2=(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)
#         inner3=SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
#         inner4=+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
#         output_tensor=inner1/inner2+inner3-inner4
#         output_tensor=torch.max(tsf.diff(high_ts,1),tsf.diff(low_ts,1))
#         return output_tensor


# class Alpha052(Functor):  #formula bug:what is L
#
#     def forword(self,high_ts,low_ts,close_ts):
#         output_tensor=tsf.rolling_sum_(MAX(0,high_ts-tsf.shift((high_ts+low_ts+close_ts)/3,1)),26)/tsf.rolling_sum_(MAX(0,tsf.shift((high_ts+low_ts+close_ts)/3,1)-L),26)*100
#         return output_tensor


class Alpha054(Functor):

    def forword(self, close_ts, open_ts):
        output_tensor = (-1 * tsf.rank(
            (tsf.rolling_std(torch.abs(close_ts - open_ts)) + (close_ts - open_ts)) + tsf.rolling_corr(close_ts,
                                                                                                       open_ts, 10)))
        return output_tensor


class Alpha055(Functor):

    def forword(self, close_ts, open_ts, high_ts, low_ts):
        inner1 = tsf.rolling_sum_(16 * (
                close_ts - tsf.shift(close_ts, 1) + (close_ts - open_ts) / 2 + tsf.shift(close_ts, 1) - tsf.shift(
            open_ts, 1)))
        cond1 = (torch.abs(high_ts - tsf.shift(close_ts, 1)) > torch.abs(low_ts - tsf.shift(close_ts, 1))) & (
                torch.abs(high_ts - tsf.shift(close_ts, 1)) > torch.abs(high_ts - tsf.shift(low_ts, 1)))
        sequence1 = torch.abs(high_ts - tsf.shift(close_ts, 1)) + torch.abs(
            low_ts - tsf.shift(close_ts, 1)) / 2 + torch.abs(tsf.shift(close_ts, 1) - tsf.shift(open_ts, 1)) / 4
        cond2 = (torch.abs(low_ts - tsf.shift(close_ts, 1)) > torch.abs(high_ts - tsf.shift(low_ts, 1))) & (
                torch.abs(low_ts - tsf.shift(close_ts, 1)) > torch.abs(high_ts - tsf.shift(close_ts, 1)))
        sequence2 = torch.abs(high_ts - tsf.shift(low_ts, 1)) + torch.abs(
            tsf.shift(close_ts, 1) - tsf.shift(open_ts, 1)) / 4
        inner2 = torch.where([~cond1 & ~cond2], sequence2, sequence1)
        output_tensor = inner1 / inner2
        return output_tensor


class Alpha056(Functor):

    def forword(self, open_ts, high_ts,low_ts,volume_ts):
        output_tensor = (tsf.rank((open_ts	-	tsf.rolling_min(open_ts,	12)))	<	tsf.rank(tsf.rank(tsf.rolling_corr(tsf.rolling_sum_(((high_ts	+	low_ts)/2),19),tsf.rolling_sum_(tsf.rolling_mean_(volume_ts,40), 19), 13))^5))
        return output_tensor


class Alpha059(Functor):

    def forword(self, close_ts,low_ts,high_ts):
        cond1= tsf.diff(close_ts,1)>0
        zeros=torch.zeros(close_ts.size())
        inner=torch.where(cond1,torch.min(low_ts, tsf.shift(close_ts, 1)),torch.max(high_ts, tsf.shift(close_ts,1)))
        cond2=tsf.diff(close_ts,1)==0
        inner=torch.where(cond2,zeros,inner)
        output_tensor=tsf.rolling_sum_(inner,20)
        return output_tensor


class Alpha049(Functor):

    def forword(self, high_ts, low_ts):
        output_tensor = torch.max(tsf.diff(high_ts, 1), tsf.diff(low_ts, 1))
        return output_tensor


class Alpha049(Functor):

    def forword(self, high_ts, low_ts):
        output_tensor = torch.max(tsf.diff(high_ts, 1), tsf.diff(low_ts, 1))
        return output_tensor
