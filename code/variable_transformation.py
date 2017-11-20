import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn import preprocessing
#%matplotlib qt
###########################################################

df = pd.read_csv('../data/df_formatted.csv')

###########################################################
# all variables to numerical by binarization

# binarization of the categorical variable Sales:
Sales_binary = pd.get_dummies(df['Sales'])
Salary_binary = pd.get_dummies(df['Salary'])

#?preprocessing.OneHotEncoder(df['Sales'])

df_binarized = df.drop(['Sales','Left','Salary'],axis=1)
df_binarized = pd.concat([df_binarized,Sales_binary,Salary_binary],axis=1,join='outer')


df_binarized.to_csv('../data/df_numerical.csv')

# normalization

min_max_scaler = preprocessing.MinMaxScaler()
df_binarized_norm = pd.DataFrame(min_max_scaler.fit_transform(df_binarized.values.astype(float)))
df_binarized_norm.columns = df_binarized.columns

###########################################################

# todo: all variables to categorical 
