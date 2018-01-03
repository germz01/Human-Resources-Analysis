import pandas as pd
import graphviz
from sklearn import tree

print 'IMPORTING THE DATA SET'

DS = pd.read_csv(filepath_or_buffer='../../data/df_formatted_ordered.csv')

print 'FORMATTING THE DATA SET'

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

for min_samples_split in [2, 0.10, 0.20]:
    print 'TRAINING GINI DECISION TREE WITH MIN_SAMPLE_SPLIT ' \
          + str(min_samples_split)

    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best',
                                      min_samples_split=min_samples_split)
    clf = clf.fit(data, target)

    print 'PLOTTING GINI DECISION TREE WITH MIN_SAMPLE_SPLIT ' \
          + str(min_samples_split)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=True,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("../../images/classification/decision_tree_gini_" +
                 str(min_samples_split))

for min_samples_split in [2, 0.10, 0.20]:
    print 'TRAINING ENTROPY DECISION TREE WITH MIN_SAMPLE_SPLIT ' \
          + str(min_samples_split)

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best',
                                      min_samples_split=min_samples_split)
    clf = clf.fit(data, target)

    print 'PLOTTING ENTROPY DECISION TREE WITH MIN_SAMPLE_SPLIT ' \
          + str(min_samples_split)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=True,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("../../images/classification/decision_tree_entropy_" +
                 str(min_samples_split))
