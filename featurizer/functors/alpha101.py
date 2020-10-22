import torch
from featurizer.interface import Functor
from featurizer.functions import time_series_functions as tsf


class Alpha006(Functor):

    def forward(self, open_ts, volume_ts):
        result = -1 * tsf.rolling_corr(open_ts, volume_ts, 10)
        return result


class Alpha009(Functor):

    def forward(self, close_ts):
        delta_close = tsf.diff(close_ts, 1)
        cond_1 = (tsf.rolling_max(delta_close, 5) > 0).squeeze()
        cond_2 = (tsf.rolling_max(delta_close, 5) < 0).squeeze()
        alpha = -1 * delta_close
        result = torch.where((cond_1 | cond_2),alpha,delta_close)
        return result


class Alpha010(Functor):

    def forward(self, close_ts):
        delta_close = tsf.diff(close_ts, 1)
        cond_1 = (tsf.rolling_max(delta_close, 4) > 0).squeeze()
        cond_2 = (tsf.rolling_max(delta_close, 4) < 0).squeeze()
        alpha = -1 * delta_close
        result = torch.where((cond_1 | cond_2),alpha,delta_close)
        return result


class Alpha9_10_customizable(Functor):

    def __init__(self,window):
        self.window = window

    def forward(self, close_ts):
        delta_close = tsf.diff(close_ts, 1)
        cond_1 = (tsf.rolling_max(delta_close, self.window) > 0).squeeze()
        cond_2 = (tsf.rolling_max(delta_close, self.window) < 0).squeeze()
        alpha = -1 * delta_close
        result = torch.where((cond_1 | cond_2),alpha,delta_close)
        return result


class Alpha012(Functor):

    def forward(self, close_ts, volume_ts):
        return torch.sign(tsf.diff(volume_ts, 1)) * (-1 * tsf.diff(close_ts, 1))


class Alpha021(Functor):

    def forward(self, close_ts, volume_ts):
        cond_1 = (tsf.rolling_mean_(close_ts, 8) + tsf.rolling_std(close_ts, 8) < tsf.rolling_mean_(close_ts, 2)).squeeze()
        cond_2 = (tsf.rolling_mean_(volume_ts, 20) / volume_ts < 1 ).squeeze()
        cond_3 = (tsf.rolling_mean_(close_ts, 8) - tsf.rolling_std(close_ts, 8) < tsf.rolling_mean_(close_ts, 2)).squeeze()
        ones = torch.ones(close_ts.size()).squeeze()
        result = torch.where ((cond_1 | (cond_2 & cond_3)),-1*ones, ones)
        return result


class Alpha023(Functor):

    def forward(self, high_ts):
        cond = (tsf.rolling_mean_(high_ts, 20) < high_ts).squeeze()
        zeros=torch.zeros(high_ts.size()).squeeze()
        result = torch.where(cond,  -1 * tsf.diff(high_ts, 2).squeeze(),zeros)
        return result


class Alpha024(Functor):

    def forward(self, close_ts):
        cond = (tsf.diff(tsf.rolling_mean_(close_ts, 100), 100) / tsf.shift(close_ts, 100) <= 0.05).squeeze()
        alpha = -1 * tsf.diff(close_ts, 3).squeeze()
        result = torch.where(cond,-1 * (close_ts - tsf.rolling_min(close_ts, 100)).squeeze(),alpha)
        return result


class Alpha024_customizable(Functor):

    def __init__(self,window):
        self.window=window

    def forward(self, close_ts):
        cond = (tsf.diff(tsf.rolling_mean_(close_ts, self.window), self.window) / tsf.shift(close_ts, self.window) <= 0.05).squeeze()
        alpha = -1 * tsf.diff(close_ts, 3).squeeze()
        result = torch.where(cond,-1 * (close_ts - tsf.rolling_min(close_ts, self.window)).squeeze(),alpha)
        return result


class Alpha046(Functor):

    def forward(self, close_ts):
        inner = ((tsf.shift(close_ts, 20) - tsf.shift(close_ts, 10)) / 10).squeeze() - ((tsf.shift(close_ts, 10) - close_ts) / 10).squeeze()
        cond_1 = inner < 0
        cond_2 = inner > 0.25
        alpha = (-1 * tsf.diff(close_ts).squeeze())
        ones=torch.ones(close_ts.size()).squeeze()
        result=torch.where((cond_1 | cond_2 ), -1*ones, alpha)
        return result


class Alpha049(Functor):

    def forward(self, close_ts):
        inner = (((tsf.shift(close_ts, 20) - tsf.shift(close_ts, 10)) / 10).squeeze() - (
                    (tsf.shift(close_ts, 10) - close_ts) / 10).squeeze()  )
        alpha = (-1 * tsf.diff(close_ts)).squeeze()
        cond=(inner < -0.1)
        ones=torch.ones(close_ts.size()).squeeze()
        result=torch.where(cond, ones, alpha)
        return result  # .values


class Alpha051(Functor):

    def forward(self, close_ts):
        inner = (((tsf.shift(close_ts, 20) - tsf.shift(close_ts, 10)) / 10) - (
                    (tsf.shift(close_ts, 10) - close_ts) / 10)).squeeze()
        alpha = (-1 * tsf.diff(close_ts)).squeeze()
        ones=torch.ones(close_ts.size()).squeeze()
        cond=(inner < -0.05).squeeze()
        result=torch.where(cond,ones,alpha)
        return result


class Alpha053(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        inner = (close_ts - low_ts).squeeze()
        constant=0.0001*torch.ones(close_ts.size()).squeeze()
        cond=inner==0
        inner=torch.where(cond,constant,inner)
        result=-1 * tsf.diff((((close_ts - low_ts) - (high_ts - close_ts)).squeeze() / inner), 9)
        return result


class Alpha054(Functor):

    def forward(self, open_ts, high_ts, low_ts, close_ts):
        inner = (low_ts - high_ts).squeeze()
        constant=0.0001*torch.ones(close_ts.size()).squeeze()
        cond=inner==0
        inner=torch.where(cond,constant,inner)
        result = -1 * ((low_ts - close_ts) * (open_ts ** 5)).squeeze() / (inner * (close_ts ** 5).squeeze())
        return result
