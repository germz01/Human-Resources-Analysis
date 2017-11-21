import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

sns.set()

DS = pd.read_csv('../HR_comma_sep.csv',
                 names=['Satisfaction_Level', 'Last_Evaluation',
                        'Number_Project', 'Average_Montly_Hours',
                        'Time_Spend_Company', 'Work_Accident', 'Left',
                        'Promotion_Last_5_Years', 'Sales', 'Salary'],
                 header=0)
LEFT = DS[DS.Left == 1]
STAYED = DS[DS.Left == 0]
MAX_K = 50

print 'PLOTTING SSE FOR EMPLOYEE WHO LEFT'

sse_list = list()

for k in range(2, MAX_K):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(LEFT[['Satisfaction_Level', 'Last_Evaluation']])
    sse = kmeans.inertia_
    sse_list.append(sse)

plt.plot(range(2, MAX_K), sse_list)
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.savefig(fname='../images/kmeans/SSE_left.pdf')

plt.clf()

print 'PLOTTING SSE FOR EMPLOYEE WHO STAYED'

sse_list = list()

for k in range(2, MAX_K):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(STAYED[['Satisfaction_Level', 'Last_Evaluation']])
    sse = kmeans.inertia_
    sse_list.append(sse)

plt.plot(range(2, MAX_K), sse_list)
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.savefig(fname='../images/kmeans/SSE_stayed.pdf')

plt.clf()

print 'PLOTTING CLUSTERS FOR EMPLOYEE WHO LEFT'

clusters = 3

kmeans = KMeans(n_clusters=clusters)
kmeans.fit(LEFT[['Satisfaction_Level', 'Last_Evaluation']])
plt.scatter(x="Satisfaction_Level", y="Last_Evaluation", data=LEFT, alpha=0.25,
            color=[sns.color_palette("YlGnBu", clusters).as_hex()[c]
                   for c in kmeans.labels_])
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
            color="black", marker="+", s=100)
plt.savefig(fname='../images/kmeans/cluster_left.pdf')

plt.clf()

print 'PLOTTING CLUSTERS FOR EMPLOYEE WHO STAYED'

clusters = 5

kmeans = KMeans(n_clusters=clusters)
kmeans.fit(STAYED[['Satisfaction_Level', 'Last_Evaluation']])
plt.scatter(x="Satisfaction_Level", y="Last_Evaluation", data=STAYED,
            alpha=0.25,
            color=[sns.color_palette("YlGnBu", clusters).as_hex()[c]
                   for c in kmeans.labels_])
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
            color="black", marker="+", s=100)
plt.savefig(fname='../images/kmeans/cluster_stayed.pdf')
