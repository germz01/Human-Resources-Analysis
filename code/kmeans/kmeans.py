import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

KMEANS_COLORS = ['#40db89', '#46d3da', '#5771d4', '#a458d4', '#e05d59',
                 '#dcc166']

sns.set()
sns.set_style("whitegrid")

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y label
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ANS = 'N'

print 'IMPORTING THE DATA SET'

DS = pd.read_csv('../../HR_comma_sep.csv',
                 names=['Satisfaction_Level', 'Last_Evaluation',
                        'Number_Project', 'Average_Montly_Hours',
                        'Time_Spend_Company', 'Work_Accident', 'Left',
                        'Promotion_Last_5_Years', 'Sales', 'Salary'], header=0)
DS = DS.drop(labels=['Work_Accident', 'Promotion_Last_5_Years', 'Sales',
             'Salary'], axis=1)

print 'SCALING THE DATA USING MIN MAX SCALER'

warnings.simplefilter("ignore")

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

scaler.fit(DS['Number_Project'].values.reshape(-1, 1))
DS['Number_Project'] = scaler.transform(DS['Number_Project'].values.
                                        reshape(-1, 1))
scaler.fit(DS['Average_Montly_Hours'].values.reshape(-1, 1))
DS['Average_Montly_Hours'] = scaler.transform(DS['Average_Montly_Hours'].
                                              values.reshape(-1, 1))
scaler.fit(DS['Time_Spend_Company'].values.reshape(-1, 1))
DS['Time_Spend_Company'] = scaler.transform(DS['Time_Spend_Company'].values.
                                            reshape(-1, 1))
DS['Average_Montly_Hours'] = [round(i, 2) for i in DS['Average_Montly_Hours']]
DS['Time_Spend_Company'] = [round(i, 2) for i in DS['Time_Spend_Company']]

kmeans = KMeans(n_clusters=4).fit(DS[['Satisfaction_Level', 'Last_Evaluation',
                                      'Number_Project', 'Average_Montly_Hours',
                                      'Time_Spend_Company']])
# DS.drop(labels=['Left'], axis=1)

# print 'silhouette', silhouette_score(DS, kmeans.labels_)

print 'CLUSTERING'

DS['Cluster'] = kmeans.labels_

fig = plt.figure(figsize=(10, 7), dpi=300)
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
patches = [mpatches.Patch(color=KMEANS_COLORS[i], label='Cluster ' + str(i))
           for i in range(0, kmeans.n_clusters)]


plt.scatter(x='Satisfaction_Level', y='Last_Evaluation', data=DS,
            color=[KMEANS_COLORS[i] for i in kmeans.labels_], s=15)
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
            color="black", marker="+", s=1000)
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('K-means applied on the whole data set')

ANS = raw_input('Do you want to save the graph?(Y/N) ')
if ANS not in ['Y', 'N']:
    print 'ERROR!!!'
elif ANS == 'Y':
    plt.savefig('../../images/kmeans/cluster_total.pdf')

print 'PLOTTING THE POPULATION DENSITY OF THE OBTAINED CLUSTERS'

fig.clf()
fig.add_axes([0.1, 0.1, 0.6, 0.75])
d = dict([(len(DS[DS.Cluster == i]), [i, KMEANS_COLORS[i]])
         for i in range(0, kmeans.n_clusters)])
keys = d.keys()
keys.sort()

for c in range(0, kmeans.n_clusters):
    k = keys.pop()
    plt.bar(x=0, height=k, color=d[k][1], label="Cluster " + str(d[k][0]))

plt.xlabel("Custers")
plt.ylabel("Employees")
plt.xticks([0], [''])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Employees per Cluster")

ANS = raw_input('Do you want to save the graph?(Y/N) ')
if ANS not in ['Y', 'N']:
    print 'ERROR!!!'
elif ANS == 'Y':
    plt.savefig('../../images/kmeans/dist_cluster.pdf')

print 'PLOTTING THE SSE GRAPH'

plt.clf()
fig, ax = plt.subplots(figsize=(15, 10), ncols=1)
ax.tick_params(axis='x', which='major', pad=15)
sse_list = list()
max_k = 21

for k in range(2, max_k):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=100)
    kmeans.fit(DS[['Satisfaction_Level', 'Last_Evaluation', 'Number_Project',
               'Average_Montly_Hours', 'Time_Spend_Company']])
    sse = kmeans.inertia_
    sse_list.append(sse)

    # print str(k) + ': ' + str(silhouette_score(DS, kmeans.labels_))

plt.plot(range(2, max_k), sse_list)
plt.xticks([2, 5, 10, 15, 20], [2, 5, 10, 15, 20])
plt.xlabel("Clusters")
plt.ylabel("SSE")

ANS = raw_input('Do you want to save the graph?(Y/N) ')
if ANS not in ['Y', 'N']:
    print 'ERROR!!!'
elif ANS == 'Y':
    plt.savefig('../../images/kmeans/SSE.pdf')


ANS = raw_input('Do you want to save the distribution?(Y/N) ')
if ANS not in ['Y', 'N']:
    print 'ERROR!!!'
elif ANS == 'Y':
        print 'SAVING THE DISTRIBUTION OF THE OBTAINED CLUSTER IN DATA FOLDER'
        DS.groupby('Cluster').describe().to_csv(
                                                '../../data/'
                                                'kmeans_distribution.csv')
        DS[DS.Left == 0].groupby('Cluster').describe().to_csv('../../data/'
                                                              'kmeans_distribu'
                                                              'tion_stayed.csv'
                                                              )
        DS[DS.Left == 1].groupby('Cluster').describe().to_csv('../../data/'
                                                              'kmeans_distribu'
                                                              'tion_left.csv'
                                                              )
