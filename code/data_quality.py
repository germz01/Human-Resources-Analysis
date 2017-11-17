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


plt.hist(np.array(df['Time_Spend_Company']))
plt.boxplot(np.array(df['Time_Spend_Company']))
plt.close()

plt.hist(np.array(df['Average_Montly_Hours']))
plt.boxplot(np.array(df['Average_Montly_Hours']))
plt.close()

plt.hist(np.array(df['Satisfaction_Level']))
plt.boxplot(np.array(df['Satisfaction_Level']))
plt.close()

plt.hist(np.array(df['Last_Evaluation']))

plt.boxplot(np.array(df['Last_Evaluation']))
plt.xlabel('prova')

plt.close()

plt.hist(np.array(df['Number_Project']))

plt.boxplot(np.array(df['Number_Project']))

plt.close()

df.query('Time_Spend_Company>6').shape[0]

df.value_counts()

plt.show()


indici = np.arange(0,5)

num_variables[indici[0]-1]



plt.figure(figsize=(10,2))
for i in indici:
    plt.subplot(1,5,i+1) 
    plt.boxplot(np.array(df[num_variables[i]]))
    plt.xlabel(num_variables[i])
    plt.tight_layout()

plt.savefig('../images/boxplots.pdf')
    
plt.close()

###########################################################

    ## grubbs test 
from  outliers import smirnov_grubbs as grubbs

prova = df.Time_Spend_Company
##prova = df.Number_Project
    
## return the data cleaned from the outliers (two-sides test)
grubbs.test(prova,alpha=0.95)

## one-side test
prob = np.arange(0.01,0.10,0.01)

outlier = list()

for i in prob:
     (len(grubbs.max_test_outliers(prova,alpha=i)))

    
outliers = grubbs.max_test_outliers(prova,alpha=0.01)

grubbs.min_test_outliers(prova,alpha=0.10)
