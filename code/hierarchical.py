import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from scipy.cluster.hierarchy import dendrogram, linkage
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

df_num = df.drop(['Left','Sales','Salary'],axis=1)

from sklearn import preprocessing
preprocessing.MinMaxScaler()

min_max_scaler = preprocessing.MinMaxScaler()


df_num_norm = min_max_scaler.fit_transform(df_num.values.astype(float))

###########################################################

methods=['ward',]

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(df_num_norm, n_neighbors=100, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)

# define the function for clustering:
ward = AgglomerativeClustering(n_clusters=2, linkage='ward', affinity='euclidean')

average_linkage = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='manhattan')

ward.fit(df_num_norm)
average_linkage.fit(df_num_norm)

ward_labels = pd.DataFrame(ward.labels_)
ward_labels_inv = abs(ward_labels-1)

average_labels = pd.DataFrame(average_linkage.labels_)

ward_labels.hist()
average_labels.hist()

import sklearn.metrics

jaccard = sklearn.metrics.jaccard_similarity_score

jaccard(ward_labels,average_labels)
jaccard(ward_labels_inv,average_labels)

jaccard(ward_labels,df.Left)
jaccard(ward_labels_inv,df.Left)

jaccard(average_labels,df.Left)

(np.array(ward_labels) & np.array(df.Left))

prova = (np.array(ward_labels_inv) & np.array(df.Left))

ward_labels_inv == df.Left



plt.close()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(np.array(ward_labels))

plt.subplot(1,2,1)
plt.hist(average_labels)



len(ward.labels_)

plt.close()


label = np.reshape(ward.labels_,face.shape)

ward.labels_

hist, bins = np.histogram(ward.labels_, bins=range(0, len(set(ward.labels_)) + 1))


plt.hist(ward.labels_)
plt.show()

print 'labels', dict(zip(bins, hist))
print 'silhouette', silhouette_score(train_data, ward.labels_)

###########################################################
# Compute clustering

print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 15  # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                               connectivity=connectivity)
ward.fit(X)
label = np.reshape(ward.labels_, face.shape)
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

###########################################################

