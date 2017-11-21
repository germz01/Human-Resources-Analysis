import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from scipy.cluster.hierarchy import dendrogram, linkage
%matplotlib qt
###########################################################

SMALL_SIZE = 20
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
# variables transformation

df = pd.read_csv('../data/df_formatted.csv')

#df_num = df.drop(['Left','Sales','Salary'],axis=1)

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

# binarization of the categorical variable Sales:
Sales_binary = pd.get_dummies(df['Sales'])
Salary_binary = pd.get_dummies(df['Salary'])

#?preprocessing.OneHotEncoder(df['Sales'])

df_binarized = df.drop(['Sales','Left','Salary'],axis=1)
df_binarized = pd.concat([df_binarized,Sales_binary,Salary_binary],axis=1,join='outer')

df_binarized_norm = pd.DataFrame(min_max_scaler.fit_transform(df_binarized.values.astype(float)))

df_binarized_norm.columns = df_binarized.columns

data = df_binarized_norm

#df_num_norm = min_max_scaler.fit_transform(df_num.values.astype(float))

###########################################################
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.neighbors import kneighbors_graph
# clustering with scipy
from scipy.spatial.distance import pdist, squareform

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples

import gc


methods = ['single','average','weighted','centroid','median','ward']
distances = ['euclidean','cityblock']

# function beginning
###########################################################

method = methods[-1]
distance = distances[0]

data_dist = pdist(data,metric = distance)
data_link = linkage(data_dist,method=method,metric=distance)

coph = cophenet(Z = data_link, Y = data_dist)
coph_corr=coph[0]

plt.close()

fig = plt.figure(figsize=(13, 5))
dn = dendrogram(Z=data_link,p=10,truncate_mode='level',orientation='left')
plt.title('Method: {} , Metric: {}, Coph Corr: {:.2f} '.format(method,distance,coph_corr))

plt.savefig('../images/hierarchical/dendrogram_{}_{}.pdf'.format(method,distance))

out = [method,distance,coph_corr]

# implementation finished
###########################################################

# silhouette analysis

labels = fcluster(data_link, 4, criterion="maxclust")
silhouette_coefficient = silhouette_score(squareform(data_dist),labels,
                                          metric='euclidean')

silhouette_labels = silhouette_samples(squareform(data_dist),labels)

df_silhouette = pd.DataFrame(np.array([labels,silhouette_labels]).T)
df_silhouette.columns = ['cluster','silhouette']
df_silhouette=df_silhouette.sort_values(['cluster','silhouette'],ascending=False)

indici = np.arange(df_silhouette.shape[0],0,step=-1)
df_silhouette = df_silhouette.assign(indici=indici)


fig = plt.figure(figsize=(10,5))

#plt.plot(x = df_silhouette['silhouette'],y=df_silhouette['indici'])
plt.scatter(x=df_silhouette['silhouette'],y=df_silhouette['indici'],c=df_silhouette['cluster'])

plt.title('Silhouette, Method: {} , Metric: {}, Coph Corr= {:.2f}, Silh. Coeff={:.2f} '.format(method,distance,coph_corr,silhouette_coefficient))

plt.axvline(x=silhouette_coefficient)

plt.savefig('../images/hierarchical/silhouette_{}_{}.pdf'.format(method,distance))


plt.close()

del(coph)
del(data_link)


gc.collect()
###########################################################
# function for hierarchical clustering

def analyze(method,distance):

    print(method)
    data_dist = pdist(data,metric = distance)
    data_link = linkage(data_dist,method=method,metric=distance)
    
    coph = cophenet(Z = data_link, Y = data_dist)
    coph_corr=coph[0]

    plt.close()

    fig = plt.figure(figsize=(13, 5))
    dn = dendrogram(Z=data_link,p=10,truncate_mode='level',orientation='left')
    plt.title('Method: {} , Metric: {}, Coph Corr: {:.2f} '.format(method,distance,coph_corr))

    plt.savefig('../images/hierarchical/dendrogram_{}_{}.pdf'.format(method,distance))
    del(coph)
    del(data_link)
    gc.collect()

out = [method,distance,coph_corr]
return out

###########################################################

## apply each method

for method in methods:
    analyze(method,distance='euclidean')


###########################################################


import sklearn.metrics

jaccard = sklearn.metrics.jaccard_similarity_score

jaccard(ward_labels,average_labels)
jaccard(ward_labels_inv,average_labels)

jaccard(ward_labels,df.Left)
jaccard(ward_labels_inv,df.Left)

jaccard(average_labels,df.Left)

