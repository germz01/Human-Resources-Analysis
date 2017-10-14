import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ds = pd.read_csv('~/Documents/Università/Magistrale/Data Mining/' +
                 'Human-Resources-Analysis/HR_comma_sep.csv')

who_left = ds[ds.left == 1]
who_stayed = ds[ds.left == 0]
columns = ['salary', 'sales', 'Work_accident', 'promotion_last_5years']

fig = plt.figure(figsize=(10, 10))
fig_dims = (2, 2)

for x in range(0, 2):
    for y in range(0, 2):
        column = columns.pop()

        plt.subplot2grid(fig_dims, (x, y))
        plt.bar(np.arange(len(ds[column].unique())),
                who_stayed[column].value_counts().tolist(),
                label='Employes who stayed')
        plt.bar(np.arange(len(ds[column].unique())),
                who_left[column].value_counts().tolist(),
                label='Employes who left')
        plt.xticks(np.arange(len(ds[column].unique())),
                   ds[column].unique())
        plt.ylabel('Employes')
        plt.legend()
        plt.title(column + ' per Employer')

plt.savefig('visualization.pdf')
