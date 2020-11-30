import inspect

import pandas as pd
import torch

import featurizer.functors.alpha192 as alpha

test_rising = torch.range(1, 100, 1)
test_falling = torch.range(100, 1, -1)
test_2d = torch.stack([test_rising, test_falling], dim=1)
result = pd.DataFrame()
for name, class_ in inspect.getmembers(alpha, inspect.isclass):
    tester = class_()
    a = inspect.signature(class_.forward)
    if len(a.parameters) == 2:
        single_case_result = tester.forward(test_2d)
    elif len(a.parameters) == 3:
        single_case_result = tester.forward(test_2d, test_2d)
    elif len(a.parameters) == 4:
        single_case_result = tester.forward(test_2d, test_2d, test_2d)
    elif len(a.parameters) == 5:
        single_case_result = tester.forward(test_2d, test_2d, test_2d, test_2d)
    elif len(a.parameters) == 6:
        single_case_result = tester.forward(test_2d, test_2d, test_2d, test_2d, test_2d)
    elif len(a.parameters) == 7:
        single_case_result = tester.forward(test_2d, test_2d, test_2d, test_2d, test_2d, test_2d)
    else:
        print(len(a.parameters))
    if name=='Alpha192':
        break
    single_case_result = single_case_result.cpu().detach().numpy()
    single_case_result = pd.DataFrame(single_case_result)
    single_case_result.columns=[name+str(i) for i in single_case_result.columns]
    result=pd.concat([result,single_case_result],axis=1)
result.to_csv('alpha192_test.csv')
