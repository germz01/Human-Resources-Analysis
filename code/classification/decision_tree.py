import pandas as pd
from sklearn import tree

DS = pd.read_csv(filepath_or_buffer='../../data/df_formatted.csv')

DS['Salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace=True)
DS['Department'].replace(['sales', 'technical', 'support', 'IT',
                          'product_mng', 'marketing', 'RandD', 'accounting',
                          'hr', 'management'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         inplace=True)

target = DS['Left'].tolist()

DS.drop(['Left'], axis=1, inplace=True)

data = DS.as_matrix()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, target)
