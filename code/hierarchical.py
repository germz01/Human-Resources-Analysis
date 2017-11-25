import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from scipy.cluster.hierarchy import dendrogram, linkage
%matplotlib qt
###########################################################

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

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

df.dtypes

df_num = df.drop(['Left','Sales','Salary','Work_Accident','Promotion_Last_5_Years'],axis=1)

df_num.shape

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


df_norm = pd.DataFrame(min_max_scaler.fit_transform(df_num.values.astype(float)))
df_norm.columns = df_num.columns
df_norm.head()


data = df_norm

#df_num_norm = min_max_scaler.fit_transform(df_num.values.astype(float))

###########################################################
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.neighbors import kneighbors_graph
# clustering with scipy
from scipy.spatial.distance import pdist, squareform

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster, cut_tree
from sklearn.metrics import silhouette_score, silhouette_samples

import gc


methods = ['single','complete','average','weighted','centroid','median','ward']
distances = ['euclidean','cityblock','mahalanobis']

# testing function for analysis
###########################################################
method = methods[2]
distance = distances[0]

data_dist = pdist(data,metric = distance)
data_link = linkage(data_dist,method=method,metric=distance)

coph = cophenet(Z = data_link, Y = data_dist)
coph_corr=coph[0]

plt.close()

fig = plt.figure(figsize=(10, 5))
dn = dendrogram(Z=data_link,p=3,truncate_mode='level',
                orientation='left',count_sort='ascendent',
                no_labels=False,show_leaf_counts=True,
)

plt.title('Method: {} , Metric: {}, Coph Corr: {:.2f} '.format(method,distance,coph_corr))

plt.savefig('../images/hierarchical/dendrogram_{}_{}.pdf'.format(method,distance))

out = [data_link,[method,distance,coph_corr,]]

# testing function finished
###########################################################

prova = np.array(cut_tree(data_link,n_clusters=3))
plt.hist(prova)

plt.close()
np.corrcoef(np.array(df.Left),prova)

jaccard(df.Left,prova)


###########################################################
# function for hierarchical clustering

def analyze(method,distance):

    print('Running method {} with metric {}'.format(method,distance))

    data_dist = pdist(data,metric = distance)
    data_link = linkage(data_dist,method=method,metric=distance)

    coph = cophenet(Z = data_link, Y = data_dist)
    coph_corr=coph[0]

    labels = fcluster(data_link, 3, criterion="maxclust")
    silhouette_coefficient = silhouette_score(squareform(data_dist),labels, metric='euclidean')


    
    plt.close()

    fig = plt.figure(figsize=(10, 5))
    dn = dendrogram(Z=data_link,p=3,truncate_mode='level',
                    orientation='left',count_sort='ascendent',
                    no_labels=False,show_leaf_counts=True,
    )

    plt.title('Method: {} , Metric: {}, Coph Corr: {:.2f} '.format(method,distance,coph_corr))

    plt.savefig('../images/hierarchical/dendrogram_{}_{}.pdf'.format(method,distance))
     
    return [method,distance,coph_corr,silhouette_coefficient]

###########################################################

## apply each method for each metric

out = []        
for method in methods:
    for distance in distances:
        out.append(analyze(method,distance=distance))


methods = ['single','complete','average','weighted','centroid','median','ward']

selected_methods = ['average','centroid']

out = []        
for method in selected_methods:
    for distance in distances:
        out.append(analyze(method,distance=distance))
        
        
import gc
gc.collect()

    
Dout =  pd.DataFrame(out)
Dout.columns = ['method','distance','coph_corr','silhouette']

Dout = Dout.sort_values(by=['coph_corr','method','distance'],ascending=False)

import seaborn as sb
plt.close()

sb.set_style('whitegrid')
plt.figure(figsize=(12,5))
sb.barplot(data=Dout, x='method',y='coph_corr',
           hue='distance')
plt.title('Comparison of different methods')
plt.tight_layout()

plt.savefig('../images/hierarchical/methods_comparison.pdf')

plt.close()

###########################################################
# silhouette comparison

plt.figure(figsize=(7,5))
sb.barplot(data=Dout, x='method',y='silhouette',
           hue='distance')
plt.title('Silhouette comparison for selected methods,two clusters')
plt.tight_layout()

plt.savefig('../images/hierarchical/silhouette_comparison.pdf')

plt.close()






###########################################################
import sklearn.metrics

jaccard = sklearn.metrics.jaccard_similarity_score

jaccard(ward_labels,average_labels)
jaccard(ward_labels_inv,average_labels)

jaccard(ward_labels,df.Left)
jaccard(ward_labels_inv,df.Left)

jaccard(average_labels,df.Left)

# silhouette analysis

labels = fcluster(data_link, 3, criterion="maxclust")
silhouette_coefficient = silhouette_score(squareform(data_dist),labels,
                                          metric='euclidean')


plt.close()
plt.hist(labels)
plt.hist(labels2)

#prova_cuttre = silhouette_coefficient


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
