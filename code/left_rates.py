import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib qt

df = pd.read_csv('../data/df_formatted.csv')

df.columns
###########################################################

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

# plt.figure(figsize=(10,4))
# plt.bar(rates_bySales.indice,[1]*len(rates_bySales.indice),
#         color='steelblue')

# plt.bar(rates_bySales.indice,rates_bySales.Left/max(rates_bySales.Left),
#         tick_label=np.array(rates_bySales.index),
#         color='tomato')
# plt.title('Leaving rate by Sales')
# plt.tight_layout()

# plt.savefig('../images/rate_bySales.pdf')

# plt.close()
###########################################################

variable = 'Sales'

def plot_rates(variable):

    by_left=df.groupby(variable).sum().Left
    counts_by=df.groupby(variable).count().Left

    rates_by=pd.DataFrame(by_left/counts_by)
    rates_by=rates_by.sort_values('Left',ascending=False)

    indice=np.arange(0,len(rates_by.index))
    rates_by=rates_by.assign(indice = indice)

    plt.close()

    plt.figure(figsize=(10,4))
    ## togli commento per grafico stacked
    ##plt.bar(rates_byvariable.indice,[1]*len(rates_byvariable.indice),
    ##        color='steelblue')

    plt.bar(rates_by.indice,rates_by.Left,
            tick_label=np.array(rates_by.index),
            color='tomato')
    plt.title('Leaving rate by variable {}'.format(variable))
    plt.tight_layout()

    plt.savefig('../images/Rates/rate_by_{}.pdf'.format(variable))

    plt.close()


df_rates = df.drop('Left',axis=1)

for variable in df_rates.columns:
    plot_rates(variable)
