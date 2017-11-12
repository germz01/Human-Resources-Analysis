import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## %matplotlib qt
##  %pylab

df = pd.read_csv('../data/df_formatted.csv')

df.head()

df.columns

plt.boxplot(np.array(df['Average_Montly_Hours']))

plt.boxplot(np.array(df['Satisfaction_Level']))

plt.boxplot(df['Satisfaction_Level'], 1)

plt.hist(df['Satisfaction_Level'])

plt.hist(df['Last_Evaluation'])

df.columns

num_variables = ['Satisfaction_Level',
                 'Average_Montly_Hours',
                 'Time_Spend_Company',
                 'Last_Evaluation',
                 'Number_Project'
]

plt.boxplot(np.array(df['Time_Spend_Company']))
plt.hist(np.array(df['Time_Spend_Company']))

plt.hist(np.array(df['Average_Montly_Hours']))
plt.boxplot(np.array(df['Average_Montly_Hours']))

plt.hist(np.array(df['Satisfaction_Level']))
plt.boxplot(np.array(df['Satisfaction_Level']))

plt.hist(np.array(df['Last_Evaluation']))
plt.boxplot(np.array(df['Last_Evaluation']))

plt.hist(np.array(df['Number_Project']))
plt.boxplot(np.array(df['Number_Project']))

df.query('Time_Spend_Company>6').shape[0]

df.value_counts()

plt.show()

plt.close()

## grubbs test 
from  outliers import smirnov_grubbs as grubbs

prova = df.Time_Spend_Company
    
## return the data cleaned from the outliers (two-sides test)
grubbs.test(prova,alpha=0.95)

## one-side test
outliers = grubbs.max_test_outliers(prova,alpha=0.10)

grubbs.min_test_outliers(prova,alpha=0.10)
