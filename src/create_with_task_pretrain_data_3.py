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

aug = pd.read_csv('inputs/xtrain_aug_es.csv')
aug = pd.merge(aug, train, on='id')[['id', 'url_legal', 'license', 'bt_excerpt', 'target', 'standard_error']]
aug.columns = ['id', 'url_legal', 'license', 'text', 'target', 'standard_error']

mlm_data_aug = aug[['text']]

new_mlm_data = pd.concat([mlm_data, mlm_data_aug], 0).reset_index(drop=True)
new_mlm_data.to_csv('inputs/mlm_data_aug.csv', index=False)
