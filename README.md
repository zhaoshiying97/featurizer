# featurizer
featurizer is a define-by-run framework for data feature engineering


## quick start
~~~
import torch
import featurizer.functors as ff

rm = ff.RollingMean(window=5)

data = torch.randn((20, 6))
rm_data = rm(data)
print(rm_data)
~~~

## install
~~~
git clone https://github.com/StateOfTheArt-quant/featurizer.git
cd featurizer
python setup.py install
~~~

## core concept: functor

A functor is class which allows an instance object of the class to be called or invoked as if it were an ordinary function. 

~~~
import pandas as pd

class Functor(object)
    def __init__(self, window=5):
        self.window = 5
    
    def forward(self, x)
        return pd.rolling_mean(x, window=5)
    
    def __call__(self, x)
        output = self.forward(x)
        return output
~~~

~~~
import numpy as np

obj = Functor(window=5)
data = np.random.random((20, 6))
output = obj(data)
~~~