
# --------------------------------------------------------------------------------------------------------- #
# -- project: Name of the Kaggle Competition                                                             -- #
# -- File: main.py | Main functionallity code                                                            -- #
# -- author: Name of the Team or the Author                                                              -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository/notebook: Private repository URL and/or public notebook in kaggle                        -- #
# -- --------------------------------------------------------------------------------------------------- -- #

import numpy as np
import pandas as pd
import functions as fn
import pickle

# -- Read input data
data_train = pd.read_csv('files/train.csv')
data_test = pd.read_csv('files/test.csv')

# drop id column
ids_train = data_train['id']
data_train.drop('id', inplace=True, axis=1)
ids_test = data_test['id']
data_test.drop('id', inplace=True, axis=1)

# estandarize variables
feats = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']

# ------------------------------------------------------------------------------------- D1: DATA SCALING -- #
mu_train, std_train = data_train[feats].mean(axis=0), data_train[feats].std(axis=0)
z_train = (data_train[feats] - mu_train)/std_train
data_train[feats] = z_train

mu_test, std_test = data_test[feats].mean(axis=0), data_test[feats].std(axis=0)
z_test = (data_test[feats] - mu_test)/std_test
data_test[feats] = z_test

# ---------------------------------------------------------------------------------- D2: DUMMY VARIABLES -- #

# One-hot encode color variable (not used when dummies are created)
# data_train['color'] = fn.variable_onehot(p_data=data_train['color'])  

# -- Dummy variables with color
data_train = pd.concat([data_train, pd.get_dummies(data_train['color'], prefix = 'color')], axis=1)
data_train = data_train.drop('color', 1)
data_test = pd.concat([data_test, pd.get_dummies(data_test['color'], prefix = 'color')], axis=1)
data_test = data_test.drop('color', 1)

# -- One-hot encode target variable
data_train['type'] = fn.variable_onehot(p_data=data_train['type'])

# -- Add Bias 
data_train['bias'] = 1
data_columns = list(data_train.columns)
data_columns.remove('bias')
data_train = data_train[['bias'] + data_columns]

data_test['bias'] = 1
data_columns = list(data_test.columns)
data_columns.remove('bias')
data_test = data_test[['bias'] + data_columns]

# -- Convert to np.array
train_data_ovr = fn.data_ovr(p_df=data_train, p_target='type')

# ------------------------------------------------------------------------------- learning based in ovr -- #  
models_ovr = fn.ovr_learning(p_data_ovr=train_data_ovr)

# model inf
models_ovr['model_2']['train']['cost']
models_ovr['model_2']['val']['cost']
models_ovr['model_0']['fitted_cost']

models_ovr['model_2']['weights']
models_ovr['model_0']['params']

# -- test
# data_test['color'] = fn.variable_onehot(p_data=data_test['color']) # Not used when dummies

# -- convert to np.array
test_data_ovr = np.array(data_test)

# -- prediction based in ovr
# vote weighting (ocurrences in train data)
oc = data_train['type'].value_counts()
vw = [np.round(oc[0]/oc.sum(), 4),
      np.round(oc[1]/oc.sum(), 4),
      np.round(oc[2]/oc.sum(), 4)]

# vw = [1, 1, 1]

# result
result = fn.ovr_predict(p_data_ovr=test_data_ovr, p_models_ovr=models_ovr, p_vote_w=vw)

# probabilistic results
result.head()

# check for balance of classes before summit results
result['decision'].value_counts()

# define experiment tag
experiment = 'submission_vfinal'

# -------------------------------------------------------------------------------------- SUBMISSION FILE -- #
submission = pd.DataFrame({'id': ids_test, 'type': result['decision']})
type_dict_sub = {0: 'Ghoul', 1: 'Goblin', 2: 'Ghost'}
submission['type'] = submission['type'].map(type_dict_sub).astype(object)
submission.to_csv('files/submissions/' + experiment + '.csv', index=False)

# ------------------------------------------------------------------------------------------ PICKLE RICK -- #
pickle_rick = 'files/submissions/' + experiment + '.dat'
with open(pickle_rick, "wb") as f:
    pickle.dump(models_ovr, f)
