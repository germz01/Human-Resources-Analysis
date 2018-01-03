import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
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

print 'IMPORTING THE DATA SET'

warnings.simplefilter("ignore")

DS = pd.read_csv('../../HR_comma_sep.csv',
                 names=['Satisfaction_Level', 'Last_Evaluation',
                        'Number_Project', 'Average_Montly_Hours',
                        'Time_Spend_Company', 'Work_Accident', 'Left',
                        'Promotion_Last_5_Years', 'Sales', 'Salary'], header=0)
LEFT, STAYED = DS[DS.Left == 1], DS[DS.Left == 0]

print 'CLUSTERING'

kmeans = KMeans(n_clusters=3).fit(LEFT[['Satisfaction_Level',
                                  'Last_Evaluation']])
LEFT['Cluster'] = kmeans.labels_

fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
patches = [mpatches.Patch(color=KMEANS_COLORS[i], label='Cluster ' + str(i))
           for i in range(0, kmeans.n_clusters)]


plt.scatter(x='Satisfaction_Level', y='Last_Evaluation', data=LEFT,
            color=[KMEANS_COLORS[i] for i in kmeans.labels_], s=40)
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
            color="black", marker="+", s=1000)
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('K-means applied on the employees who left')
# plt.show()
plt.savefig('../../images/kmeans/cluster_left.pdf')

plt.clf()

kmeans = KMeans(n_clusters=5).fit(STAYED[['Satisfaction_Level',
                                  'Last_Evaluation']])
STAYED['Cluster'] = kmeans.labels_

fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
patches = [mpatches.Patch(color=KMEANS_COLORS[i], label='Cluster ' + str(i))
           for i in range(0, kmeans.n_clusters)]


plt.scatter(x='Satisfaction_Level', y='Last_Evaluation', data=STAYED,
            color=[KMEANS_COLORS[i] for i in kmeans.labels_], s=40)
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
            color="black", marker="+", s=1000)
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('K-means applied on the employees who stayed')
# plt.show()
plt.savefig('../../images/kmeans/cluster_stayed.pdf')
