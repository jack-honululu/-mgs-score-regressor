# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import os
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt


#%%
main = pd.read_csv("../data/black_mice_labels/v2_main.csv").set_index('index')
#%%
tar = pd.read_csv("../data/black_mice_labels/v2_pain.csv").set_index('index')
#%%
invalid_data = tar.loc[tar['pain']==-1]
main = main.drop(invalid_data.index)
tar = tar.drop(invalid_data.index)
#%%
experiment = main['experiment']
experiment_set = experiment.unique()
#%%
for i in experiment_set:
    subset = main.loc[main['experiment'] == i]
    mice_id = subset['id'].unique()
    test_index = []

    if i == 'IN' or i == 'KXN':
        mice_id = np.random.choice(mice_id, 6)
        test = subset.loc[subset['id'].isin(mice_id)]
        test_index = test.index
        test_file = [os.path.join("./black_mice/black_mice/raw_set",i) for i in test_index ]
        np.save( "test_" + i + ".npy", test_file)
        test_tar = tar.loc[test_index]
        test_tar_file = np.array(test_tar['pain'])
        np.save( "test_tar_"+ i + ".npy", test_tar_file )

    train = subset.drop(test_index)
    train_index = train.index
    train_file = [os.path.join("./black_mice/black_mice/raw_set",i) for i in train_index ]
    np.save( "train_" + i + ".npy",train_file )
    train_tar = tar.loc[train_index]
    train_tar_file = np.array(train_tar['pain'])
    np.save( "train_tar_"+ i + ".npy", test_tar_file )
    print(len(train_file))


# %%


# %%
