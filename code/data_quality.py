import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib qt
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

variables = ['Satisfaction_Level',
                 'Average_Montly_Hours',
                 'Time_Spend_Company',
                 'Last_Evaluation',
                 'Number_Projects'
]

variables_units = {'Satisfaction_Level':'0-1 interval',
                   'Average_Montly_Hours':'',
                   'Time_Spend_Company':'Years',
                   'Last_Evaluation':'0-1 interval',
                   'Number_Projects':''}


fig,ax =plt.subplots()
ax.plot([1, 2, 3])
ax.legend(['A simple line'])

ax.show()


df_num = df[num_variables]

variable="Time_Spend_Company"
###########################################################

plt.close()

variable= variables[1]

fig = plt.figure(figsize=(15,4))

for variable in variables:

    indice = (variables.index(variable)+1) # indice variabile
    ax =plt.subplot(1,5,indice) 

    # aggiunta dei dati ai boxplot
    y = df[variable]
    delta = 0.3
    x = np.random.uniform(1-delta,1+delta , size=len(y))
    #plt.plot(x, y, 'b.', alpha=0.005)
   
    # boxplots
    linewidth=2 # 
    bp = ax.boxplot(np.array(df[variable]),notch=False,
                    vert=True,meanline=True,showmeans=True,
                    widths=0.5,
                    medianprops = dict(linestyle='--', linewidth=linewidth, color='firebrick'),
                    meanprops = dict(linestyle=':', linewidth=linewidth, color='orange')                
    )

    if (indice==5):
        plt.ylim(df[variable].min()-df[variable].min()*0.2
                 ,1.2*df[variable].max())
        plt.legend([bp['means'][0],bp['medians'][0]],['media','mediana'],loc=1)

    #plt.title("Boxplot of "+variable)
    plt.title(variable)

    
    if variables_units[variable] == "":
        plt.ylabel("{}".format(variable))
    else:
        plt.ylabel("{} [{}]".format(variable,variables_units[variable]))

    plt.xticks([],[])
    plt.tight_layout()

plt.savefig('../images/boxplots.pdf')


###########################################################          
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
