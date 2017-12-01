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
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y label
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('titlesize', titlesize=BIGGER_SIZE)  # fontsize of the figure title



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


data = df_norm

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

inconsistency = inconsistent(data_link,d=2)

prova = h.maxinconsts(data_link,inconsistency)

plt.hist(prova)

plt.close()

fig = plt.figure(figsize=(10, 5))
dn = dendrogram(Z=data_link,p=7,truncate_mode='level',
                orientation='left',count_sort='ascendent',
                no_labels=False,show_leaf_counts=True,
)

plt.title('Method: {} , Metric: {}, Coph Corr: {:.2f} '.format(method,distance,coph_corr))

plt.savefig('../images/hierarchical/dendrogram_{}_{}.pdf'.format(method,distance))

out = [data_link,[method,distance,coph_corr,]]

# testing function finished
###########################################################


###########################################################
# function for hierarchical clustering

def analyze(method,distance):

    print('Running method {} with metric {}'.format(method,distance))

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
     
    return [method,distance,coph_corr]

###########################################################

## apply each method for each metric

methods = ['single','complete','average','weighted','centroid','median','ward']

out = []        
for method in methods:
    for distance in distances:
        out.append(analyze(method,distance=distance))

###########################################################        
# which is the better method?
# Let's compare the cophenetic correlation

Dout =  pd.DataFrame(out)
Dout.columns = ['method','distance','coph_corr']

Dout = Dout.sort_values(by=['coph_corr','method','distance'],ascending=False)



import seaborn as sb
plt.close()


current_palette = sns.color_palette('deep')
sns.palplot(current_palette)

colors = ["windows blue", "amber", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))

mypalette = sns.color_palette('husl',3)
sns.palplot(mypalette)

sns.set_palette(palette = 'huls',3)

sb.xkcd_rgb.keys()

colors = ["olive yellow","windows blue", "orange red"]

palette = sns.xkcd_palette(colors)
sb.palplot(palette)
sb.set_palette(palette)

sb.set_style('whitegrid')

plt.close()
plt.figure(figsize=(16,5))

sb.barplot(data=Dout, x='method',y='coph_corr',
           hue='distance')

plt.legend(bbox_to_anchor=(1, 1), borderaxespad=0.)
plt.tight_layout()
plt.xlabel('Method')
plt.ylabel('Cophenetic correlation')
plt.title('Comparison of different methods')
plt.tight_layout(pad=1)

plt.savefig('../images/hierarchical/methods_comparison.pdf')
plt.close()


from ggplot import *
ggplot(aes(weight='coph_corr',x='method'),data=Dout)+geom_bar(stat='identity')


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

n_clusters = 3

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


calc_silhouette(2)



###########################################################
###########################################################
###########################################################
import sklearn.metrics

n_clusters = 2

cluster_labels = fcluster(data_link, (n_clusters), criterion="maxclust")

jaccard = sklearn.metrics.jaccard_similarity_score

df.Left.hist()

plt.hist(cluster_labels)

jaccard()


plt.close()

jaccard(ward_labels,average_labels)
jaccard(ward_labels_inv,average_labels)

jaccard(ward_labels,df.Left)
jaccard(ward_labels_inv,df.Left)

jaccard(average_labels,df.Left)



