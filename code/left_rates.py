import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib qt

df = pd.read_csv('../data/df_formatted.csv')

df.columns

bySales_left=df.groupby('Sales').sum().Left
counts_sales=df.groupby('Sales').count().Left

rates_bySales=pd.DataFrame(bySales_left/counts_sales)
rates_bySales=rates_bySales.sort_values('Left',ascending=False)

indice=np.arange(0,len(rates_bySales.index))
rates_bySales=rates_bySales.assign(indice = indice)

plt.figure(figsize=(10,4))
## togli commento per grafico stacked
##plt.bar(rates_bySales.indice,[1]*len(rates_bySales.indice),
##        color='steelblue')

plt.bar(rates_bySales.indice,rates_bySales.Left,
        tick_label=np.array(rates_bySales.index),
        color='tomato')
plt.title('Leaving rate by Sales')
plt.tight_layout()

plt.savefig('../images/rate_bySales.pdf')

plt.close()
###########################################################

plt.figure(figsize=(10,4))
## togli commento per grafico stacked
plt.bar(rates_bySales.indice,[1]*len(rates_bySales.indice),
        color='steelblue')

plt.bar(rates_bySales.indice,rates_bySales.Left/max(rates_bySales.Left),
        tick_label=np.array(rates_bySales.index),
        color='tomato')
plt.title('Leaving rate by Sales')
plt.tight_layout()

plt.savefig('../images/rate_bySales.pdf')

plt.close()
