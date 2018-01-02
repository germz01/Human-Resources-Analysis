import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score

DS = pd.read_csv(filepath_or_buffer='../../data/HR_comma_sep.csv')

feature_names = DS.columns.tolist()
feature_names.pop()

DS['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace=True)
DS['sales'].replace(['sales', 'technical', 'support', 'IT',
                     'product_mng', 'marketing', 'RandD', 'accounting',
                     'hr', 'management'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    inplace=True)

target = DS['left'].tolist()

DS.drop(['left'], axis=1, inplace=True)

train = DS.as_matrix()

csvfile = open('../../data/classification.csv', 'wb')
fieldnames = ['Criterion', 'Min Sample Split', 'Len Train', 'Len Test',
              'Accuracy', 'Precision', 'Recall']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

for criterion in ['gini', 'entropy']:
    for min_sample_split in [2, 0.1, 0.2]:
        print '\nTRAINING DECISION TREE WITH CRITERION ' + criterion + \
              ' AND MIN SAMPLE SPLIT ' + str(min_sample_split)

        clf = tree.DecisionTreeClassifier(criterion=criterion, splitter='best',
                                          max_depth=None,
                                          min_samples_split=min_sample_split,
                                          min_samples_leaf=2)
        clf = clf.fit(train, target)

        print 'PREDICTION USING TRAINING SET AS TEST SET'

        pred_target = clf.predict(train)

        pre = metrics.precision_score(target, pred_target)
        rec = metrics.recall_score(target, pred_target)
        acc = metrics.accuracy_score(target, pred_target)

        cm = confusion_matrix(target, pred_target)

        plt.matshow(cm)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(fname='../../images/classification/confusion_matrices/' +
                    criterion + '_confusion_matrix_' + 'all_test.pdf',
                    format='pdf', bbox_inches='tight')
        plt.clf()

        writer.writerow({fieldnames[0]: criterion,
                        fieldnames[1]: min_sample_split,
                        fieldnames[2]: len(train),
                        fieldnames[3]: len(train),
                        fieldnames[4]: round(acc, 2),
                        fieldnames[5]: round(pre, 2),
                        fieldnames[6]: round(rec, 2)})

        for test_size in [0.20, 0.30, 0.40]:
            print 'PREDICTION USING ' + str(1 - test_size) + '% RECORDS AS ' \
                  'TRAINING AND ' + str(test_size) + '% AS TESTING'

            train_x, test_x, train_y, test_y = train_test_split(
                                                train, target,
                                                test_size=test_size,
                                                random_state=0)
            clf = clf.fit(train_x, train_y)
            test_pred = clf.predict(test_x)

            pre = metrics.precision_score(test_y, test_pred)
            rec = metrics.recall_score(test_y, test_pred)
            acc = metrics.accuracy_score(test_y, test_pred)

            cm = confusion_matrix(test_y, test_pred)

            plt.matshow(cm)
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            plt.savefig(fname='../../images/classification/confusion_matrices/'
                        + criterion + '_confusion_matrix_' + str(test_size) +
                        '_test.pdf', format='pdf', bbox_inches='tight')
            plt.clf()

            writer.writerow({fieldnames[0]: criterion,
                            fieldnames[1]: min_sample_split,
                            fieldnames[2]: len(train_x),
                            fieldnames[3]: len(test_x),
                            fieldnames[4]: round(acc, 2),
                            fieldnames[5]: round(pre, 2),
                            fieldnames[6]: round(rec, 2)})

csvfile.close()

cfile = pd.read_csv(filepath_or_buffer='../../data/classification.csv')
print cfile.to_latex()
