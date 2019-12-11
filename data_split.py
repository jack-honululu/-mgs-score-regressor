# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import os
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from PIL import Image
from copy import deepcopy 

#%%

## read train and target csv files
main = pd.read_csv("../data/black_mice_labels/v2_main.csv")
tar = pd.read_csv("../data/black_mice_labels/v2_pain.csv")
mgs = pd.read_csv("../data/black_mice_labels/v2_mgs_00.csv")

## drop off invalid data
invalid_data = tar.loc[tar['pain']==-1]
main = main.loc[main['index'].isin(invalid_data['index'])==False]
mgs = mgs[mgs['ot1'] != '-']
# for experiments
KXN = main[main['experiment']== 'KXN']

KXN_mgs = mgs.loc[mgs['index'].isin(KXN['index'])]
KXN_pain = tar.loc[tar['index'].isin(KXN['index'])]
##test data+ part of KXN_classifier data consists classifier test 
test_data3 = KXN_pain.loc[KXN_pain['index'].isin(KXN_mgs['index'])==True]
KXN_pain = KXN_pain.loc[KXN_pain['index'].isin(KXN_mgs['index'])==False]
KXN_classifier_train_tar = deepcopy(KXN_pain)
test_data1 = KXN_pain.loc[KXN_pain['pain']==1]
test_data1 = test_data1.sample(568)
test_data2 = KXN_pain.loc[KXN_pain['pain']==0]
test_data2 = test_data2.sample(568)

KXN_pain = KXN_pain.drop(test_data1.index)
KXN_pain = KXN_pain.drop(test_data2.index)

KXN_classifier_test = pd.concat([test_data1, test_data2, test_data3])
KXN_classifier_test = KXN_classifier_test[['index','pain']].reset_index(drop=True)
KXN_classifier_test.to_csv('./KXN_classifier_test.csv',index=False)

## KXN_classifier data
KXN_classifier_tar = KXN_classifier_train_tar.reset_index(drop=True)
KXN_classifier_tar.to_csv('./KXN_classifier_tar.csv',index=False)

KXN_mgs = KXN_mgs.reset_index(drop=True)
KXN_mgs.to_csv('./KXN_mgs_tar.csv',index=False)
# %%
x= pd.read_csv('./KXN_classifier.csv')
x
# %%
from sklearn.model_selection import StratifiedKFold
data = pd.read_csv('./KXN_classifier_tar.csv')
kf = StratifiedKFold(n_splits=9, random_state=123, shuffle=True)

for y, x in kf.split(data, data['pain']):
    print(len(y),len(x))
    df = data.iloc[x]
    print(df.groupby('pain')['index'].nunique())

# %%
df = data.iloc[x]
df.groupby('pain')['index'].nunique()

# %%
