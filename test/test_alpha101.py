import torch
import unittest
import featurizer.functors.alpha101 as alpha
import pandas as pd

test_rising = torch.range(1,100,1)
test_falling = torch.range(100,1,-1)
test_2d = torch.stack([test_rising,test_falling],dim=1)
tester=alpha.Alpha003()
print(tester.forward(test_2d,test_2d))
tester2=alpha.Alpha006()
print(tester2.forward(test_2d,test_2d))
