import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import Kmeans

sns.set()

DS = pd.read_csv('../HR_comma_sep.csv',
                 names=['Satisfaction_Level', 'Last_Evaluation',
                        'Number_Project', 'Average_Montly_Hours',
                        'Time_Spend_Company', 'Work_Accident', 'Left',
                        'Promotion_Last_5_Years', 'Sales', 'Salary'],
                 header=0)
LEFT = DS[DS.Left == 1]
STAYED = DS[DS.Left == 0]

kmeans = Kmeans(n_clusters=3)
kmeans.fit(LEFT[['Satisfaction_Level', 'Last_Evaluation']])
kmeans_colors = [
    'green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_
    ]

plt.scatter(x="Satisfaction_Level", y="Last_Evaluation", data=LEFT, alpha=0.25,
            color=kmeans_colors)
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
            color="black", marker="X", s=100)
plt.show()
