
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
%matplotlib qt

df = pd.read_csv('../data/df_formatted.csv')

left_counts = df.Left.value_counts()

tot = df.shape[0]
left_perc = round(left_counts/tot,2)*100


assex = 0

plt.figure(figsize=(3,5))
plt.bar(assex,left_counts[0],color='steelblue')
plt.bar(assex,left_counts[1],color='tomato')
plt.title('Employees who Left')
plt.annotate(' Stayed {:.0f} %'.format(left_perc[0]), xy=(0,2000),xytext=(0,5000))
plt.annotate(' Left {:.0f} %'.format(left_perc[1]), xy=(0,1000),xytext=(0,1000))

plt.tight_layout()

plt.savefig('../images/obiettivi.pdf')

plt.close()
