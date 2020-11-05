import torch
import unittest
import featurizer.functors.alpha192 as alpha
import inspect

class TestAlpha192(unittest.TestCase):

    def  setUp(self) -> None:
        test_rising = torch.range(1, 1000, 1)
        test_falling = torch.range(1000, 1, -1)
        self.test_2d = torch.stack([test_rising, test_falling], dim=1)
        for name, class_ in inspect.getmembers(alpha, inspect.isclass):
            print(name, class_)


    def test_Alpha001(self):
        pass


for name, class_ in inspect.getmembers(alpha, inspect.isclass):
    a=inspect.signature(class_.forward)
    print(name, class_)