import sys
sys.path.append("/root/workspace/CommonLitReadabilityPrize")

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


import pandas as pd
import numpy as np

train = pd.read_csv("inputs/train.csv")
test = pd.read_csv("inputs/test.csv")

mlm_data = train[['excerpt']]
mlm_data = mlm_data.rename(columns={'excerpt':'text'})
mlm_data.to_csv('inputs/mlm_data.csv', index=False)

mlm_data_val = test[['excerpt']]
mlm_data_val = mlm_data_val.rename(columns={'excerpt':'text'})
mlm_data_val.to_csv('inputs/mlm_data_val.csv', index=False)

new_mlm_data = pd.concat([mlm_data, mlm_data_val], 0).reset_index(drop=True)
new_mlm_data.to_csv('inputs/mlm_data_all.csv', index=False)
