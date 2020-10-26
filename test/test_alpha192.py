import featurizer.functors.alpha192 as alpha
import torch
a=torch.tensor([1.0,0.0,-1.0])
b=torch.ones(a.size())*2
print(abs(a))