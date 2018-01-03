import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from scipy.cluster.hierarchy import dendrogram, linkage
%matplotlib qt
###########################################################

SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y label
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#plt.rc('titlesize', titlesize=BIGGER_SIZE)  #fontsize of the figure title
###########################################################
# variables transformation

df = pd.read_csv('../data/df_formatted.csv')

df.dtypes

df_num = df.drop(['Left','Department','Salary','Work_Accident','Promotion_Last_5_Years'],axis=1)

df_num.dtypes
df_num.shape

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


df_norm = pd.DataFrame(min_max_scaler.fit_transform(df_num.values.astype(float)))
df_norm.columns = df_num.columns
df_norm.head()


from sklearn.preprocessing import StandardScaler

df_z = StandardScaler().fit_transform(df_num.values)
df_z = pd.DataFrame(df_z)
df_z.columns = df_num.columns
df_z.head()




data = df_z

#df_num_norm = min_max_scaler.fit_transform(df_num.values.astype(float))

###########################################################
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.neighbors import kneighbors_graph
# clustering with scipy
from scipy.spatial.distance import pdist, squareform

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster, cut_tree, inconsistent

import scipy.cluster.hierarchy as h

from sklearn.metrics import silhouette_score, silhouette_samples

import gc
###########################################################

selected_methods = ['centroid','average']

distances = ['euclidean']
distance='euclidean'

cut_levels = [3.2,3.6]

#for i,method in enumerate(selected_methods):

method = 'centroid'
data_dist = pdist(data,metric = distance)
data_link = linkage(data_dist,method=method,metric=distance)
    
lista_nclusters = [2,3,4,5,6]
lista_silhouette_centroid = []

for n_clusters in lista_nclusters:
    cluster_labels = fcluster(data_link, (n_clusters), criterion="maxclust")
    print n_clusters
    lista_silhouette_centroid.append(silhouette_score(squareform(data_dist),cluster_labels, metric='euclidean'))
    print str(n_clusters)+" finito"


    
labels = pd.DataFrame(cluster_labels)
labels.describe()
###########################################################

method = 'average'
data_dist = pdist(data,metric = distance)
data_link = linkage(data_dist,method=method,metric=distance)

lista_silhouette_average = []

for n_clusters in lista_nclusters:
    cluster_labels = fcluster(data_link, (n_clusters), criterion="maxclust")
    print n_clusters
    lista_silhouette_average.append(silhouette_score(squareform(data_dist),cluster_labels, metric='euclidean'))
    print lista_silhouette_average
    print str(n_clusters)+" finito"


plt.close()    
fig = plt.fig(figsize=(7,7))
plt.plot(lista_nclusters, lista_silhouette_centroid, label="centroid")
plt.plot(lista_nclusters, lista_silhouette_average, label="average")
plt.scatter(lista_nclusters, lista_silhouette_centroid)
plt.scatter(lista_nclusters, lista_silhouette_average)
plt.title("Silhouette al variare del taglio del dendrogramma")
plt.grid()
plt.xlabel("Numero di cluster")
plt.ylabel(" Silhouette Media")
plt.tight_layout()
plt.legend()
plt.savefig('../images/hierarchical/confronto_silhouette.pdf')

# i due metodi sono quasi equivalenti, per confrontare con
# k-means uso average

###########################################################
# guardo il numero di valori dentro i clusters

method = 'centroid'
data_dist = pdist(data,metric = distance)
data_link = linkage(data_dist,method=method,metric=distance)

lista_risultati= []
lista_nomi = []

n_clusters=2

def cluster_count(n_clusters):

    method = 'average'
    data_dist = pdist(data,metric = distance)
    data_link = linkage(data_dist,method=method,metric=distance)
    average_labels = fcluster(data_link,(n_clusters), criterion="maxclust")

    method = 'centroid'
    data_dist = pdist(data,metric = distance)
    data_link = linkage(data_dist,method=method,metric=distance)
    centroid_labels = fcluster(data_link,(n_clusters), criterion="maxclust")

    Dlabels2 = pd.DataFrame([average_labels,centroid_labels]).transpose()
    Dlabels2.columns = ['average','centroid']


    Dcount2=pd.DataFrame([Dlabels2.average.value_counts(),
                         Dlabels2.centroid.value_counts()]
    )
    Dcount2.columns = ["cluster{}".format(i) for i in range(n_clusters)]

    return(Dcount2)

Dcount2 = cluster_count(2)
Dcount3 = cluster_count(3)
Dcount4 = cluster_count(4)

Dcount5 = cluster_count(5)

Dcount2.to_latex("../data/hierarchical/Dcount2.txt")
Dcount3.to_latex("../data/hierarchical/Dcount3.txt")
Dcount4.to_latex("../data/hierarchical/Dcount4.txt")
Dcount5.to_latex("../data/hierarchical/Dcount5.txt")



###########################################################
# from the selected method which are the best?
# where to cut the dendrogram?
# i.e which is the optimal number of clusters?
# -> silhouette comparison
import matplotlib.cm as cm

selected_methods = ['average','centroid']

out = []        
for method in selected_methods:
    for distance in distances:

        
method = 'centroid'        
distance = 'euclidean'        

data_dist = pdist(data,metric = distance)
data_link = linkage(data_dist,method=method,metric=distance)

#coph = cophenet(Z = data_link, Y = data_dist)
#coph_corr=coph[0]

n_clusters = 4

def calc_silhouette(n_clusters):
    
    cluster_labels = fcluster(data_link, (n_clusters+1), criterion="maxclust")
    silhouette_average = silhouette_score(squareform(data_dist),cluster_labels, metric='euclidean')
    silhouette_values = silhouette_samples(squareform(data_dist),cluster_labels)

    plt.close()

    # Create a subplot with 1 row and 2 columns
    fig, ax1  = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)

    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(silhouette_data)*1.05])

    y_lower = 0
    for i in range(1,n_clusters+1):
    # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_silhouette_values = silhouette_values[cluster_labels == i]

        ith_silhouette_values.sort()

        size_cluster_i = ith_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.9, y_lower + 0.5 * size_cluster_i,
                 'Cluster {}'.format(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 0  # 10 for the 0 samples

    #    ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_average, color="red", linestyle="--")

        #ax1.set_yticks([])  # Clear the yaxis labels / ticks

        ax1.set_xticks(np.arange(-1,1.2,0.2))


    plt.suptitle("Silhouette for method {} and metric {} , with {} clusters".format(method,distance, n_clusters),fontsize=14, fontweight='bold')

    #plt.tight_layout()

    plt.savefig('../images/hierarchical/silhouette_{}_{}_n{}.pdf'.format(method,distance,n_clusters))

    return silhouette_average

plt.close()


for n_clusters in range(2,5):
    print(n_clusters)
    calc_silhouette(n_clusters)


calc_silhouette(4)




# testing function for analysis
###########################################################
selected_methods = ['centroid','average']



#method =selected_methods[0]
#distance = 'cityblock'
for method in selected_methods:
    for distance in distances:

        data_dist = pdist(data,metric = distance)
        data_link = linkage(data_dist,method=method,metric=distance)

        ##coph = cophenet(Z = data_link, Y = data_dist)
        ##coph_corr=coph[0]
        #coph_corr

        ##inconsistency = inconsistent(data_link,d=2)
        ##prova = h.maxinconsts(data_link,inconsistency)
        #plt.hist(prova)

        cut_level = max(data_link[:,2])*0.8

        plt.close()
        fig = plt.figure(figsize=(10, 5))
        dn = dendrogram(Z=data_link,##truncate_mode='level', p=20,
                        orientation='top',
         ##               count_sort='descending',
         ##              distance_sort='descending',
                        no_labels=True,show_leaf_counts=True,
                        color_threshold = cut_level
        )
        plt.axhline(y=cut_level,linestyle='--',
                    color='black')

        plt.title('Dendrogram for method "{}" and distance "{}"'.format(method,distance))
        plt.ylabel("Distance")
        #plt.xlabel("")
        plt.savefig('../images/hierarchical/dendrogram_{}_{}.pdf'.format(method,distance))

        
