import pandas as pd
import graphviz
from sklearn import tree

DS = pd.read_csv(filepath_or_buffer='../../data/df_formatted_ordered.csv')
feature_names = DS.columns.tolist()
feature_names.pop()

DS['Salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace=True)
DS['Department'].replace(['sales', 'technical', 'support', 'IT',
                          'product_mng', 'marketing', 'RandD', 'accounting',
                          'hr', 'management'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         inplace=True)

target = DS['Left'].tolist()

DS.drop(['Left'], axis=1, inplace=True)

data = DS.as_matrix()

clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best',
                                  min_samples_split=0.10)
clf = clf.fit(data, target)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=True,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("../../images/classification/decision_tree_gini")

clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')

clf = clf.fit(data, target)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=True,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("../../images/classification/decision_tree_entropy")
