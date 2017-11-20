
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

%matplotlib qt

###########################################################

#SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y label
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
###########################################################


df = pd.read_csv('../data/df_formatted.csv')

df.head()

left_counts = df.Left.value_counts()

tot = df.shape[0]
left_perc = round(left_counts/tot,2)*100
assex = 0

## old plot

plt.figure(figsize=(3,5))
plt.bar(assex,tot,color='steelblue')
plt.bar(assex,left_counts[1],color='tomato')
plt.title('Employees who Left')
plt.annotate(' Stayed {:.0f} %'.format(left_perc[0]), xy=(0,2000),xytext=(0,5000))
plt.annotate(' Left {:.0f} %'.format(left_perc[1]), xy=(0,1000),xytext=(0,1000))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

plt.yticks([left_counts[1],tot],)
###########################################################
# new plot

plt.close()

fig, ax1 = plt.subplots(figsize=(5,6.5))

ax1.bar(assex, tot,color='steelblue')
#ax1.set_xlabel('Empl')

# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Number of employees')
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

ax1.bar(assex, left_counts[1],color='tomato')
ax1.set_ylim([0,tot])
ax1.set_yticks([0,left_counts[1],math.ceil(tot/2),
                math.ceil(tot*0.75),
                tot],)

#plt.title('Employees who left')

ax2 = ax1.twinx()
#ax2.set_ylabel('Percentage')
#ax2.tick_params('y', colors='r')

ax2.set_ylim([0,100])
percentuali = [0,24,50,75,100]
ylab = [str(perc)+'%' for perc in percentuali]
ax2.set_yticks(percentuali)
ax2.set_yticklabels(ylab)



ax2.grid()

#ax1.legend([1,2],[1,2])
#axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)

fig.tight_layout()

plt.savefig('../images/obiettivi.pdf')


#plt.close()



