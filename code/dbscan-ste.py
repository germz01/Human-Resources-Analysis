import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

###########################################################
# constants for plot sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y label
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('titlesize', titlesize=BIGGER_SIZE)  # fontsize of the figure title
###########################################################
# data pre-processing

df = pd.read_csv('../data/df_formatted.csv')

df.dtypes

df_num = df[['Satisfaction_Level','Last_Evaluation',
            'Average_Montly_Hours',
             'Time_Spend_Company','Number_Projects']]
df_num.dtypes

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

df_mmax = pd.DataFrame(min_max_scaler.fit_transform(df_num.values.astype(float)))
df_mmax.columns = df_num.columns
df_mmax.head()

from sklearn.preprocessing import StandardScaler

df_z = StandardScaler().fit_transform(df_num.values)
df_z = pd.DataFrame(df_z)
df_z.columns = df_num.columns
df_z.head()

# quale normalizzazione utilizzare? proseguo con quella standard

data = df_z

###########################################################

# scelta di un intervallo per eps e MinPts tramite k-NN

from sklearn.neighbors import NearestNeighbors

#data = df_mmax
data = df_z

# grafico k-nearest-neighbors
plt.close()
plt.figure(figsize=(12,4))

for k in range(3,20,1): # k =  numero di vicini

    nbrs = NearestNeighbors(n_neighbors=k, metric='minkowski', p=2).fit(data.values) # uso distanza euclidea

    distances, indices = nbrs.kneighbors(data.values)
    df_kdist = pd.DataFrame(distances)
    df_kdist.columns = [str(i) for i in range(1,k+1)]
    df_kdist = df_kdist.sort_values(str(k))

    kdist = df_kdist[str(k)] # seleziono massima distanza tra i k-vicini

    x=range(1,15000-1)
    plt.scatter(x=range(1,15000),y=(kdist),s=3)
    
plt.title('Nearest Neighbors')
plt.ylabel('k-distances for k nearest neighbors')
plt.xlabel('Data objects sorted by k-distances')

plt.axhline(1.2)
plt.axhline(0.8)
plt.tight_layout()

""" grafico da sistemare, serve una legenda per il valore di k,
usare linee invece che punti
"""

plt.savefig('../images/dbscan/nearestneighbors.pdf')

plt.close()

###########################################################
""" dal grafico di k-neighbors è stato ricavato un range per 
eps che viene raffinato plottando il numero di clusters risultanti
in funzione di eps e MinPts """

from sklearn.cluster import DBSCAN


iter_eps = np.linspace(0.86,1.02,40) 
iter_MinPts = range(4,20,1)

l=len(iter_eps)*len(iter_MinPts)
l
# ci mette un pò con queste impostazioni.. 
# inizialmente provare con un numero minore di eps/MinPts

risultati = []
###########################################################
i=0 # contatore per stato avanzamento del ciclo
for MinPts in iter_MinPts:
    for eps in iter_eps:
        out = DBSCAN(eps=eps,min_samples = MinPts,metric='euclidean',
                     algorithm='auto',n_jobs=-1).fit(data)
        labels = out.labels_
# determino numero di cluster da labels
# prendo elementi unici e rimuovo il rumore (label=-1)
        labels_unique = list(set(labels))

        if (-1)  in labels_unique:
            labels_unique.remove(-1)
            
        nc= len(labels_unique) # numero di clusters
        del(labels)
        del(labels_unique)
        risultati.append([nc,MinPts,eps])
        i+=1
        print(str(i)+'/'+str(l))

###########################################################

df_risultati = pd.DataFrame(risultati)
df_risultati.columns=['nc','MinPts','eps']
#df_risultati.sort_values('nc')
df_risultati.eps = df_risultati.eps.round(3)

#df_filtered = df_risultati[df_risultati.eps<0.1]
#df_risultati.sort_values()
#df_risultati[df_risultati.nc<10] = 10

df_pivot = df_risultati.pivot('eps','MinPts','nc')
#df_pivot = df_filtered.pivot('eps','MinPts','nc')

plt.close()
plt.figure(figsize=(15,7))
sns.heatmap(df_pivot,annot=True,robust=True,square=False,center=4,cmap="YlGnBu",)
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.title('Numero di clusters in funzione dei parametri')
plt.tight_layout()

plt.savefig('../images/dbscan/heatmap.pdf')

###########################################################
""" scelta finale dei parametri, in base alle zona più 
larga e stabile nell'heatmap
"""
###########################################################
# ad esempio per 4 clusters:
eps = 0.940
MinPts = 7

out = DBSCAN(eps=eps,min_samples=MinPts,metric='euclidean',
             algorithm='auto',n_jobs=-1).fit(df_z)
labels = out.labels_

prova = pd.Series(labels)
prova.value_counts()

###########################################################
