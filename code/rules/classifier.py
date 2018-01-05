import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.metrics import confusion_matrix
from sklearn import metrics

sns.set()
sns.set_style("whitegrid")
FIG = plt.figure(figsize=(8, 5))

ds = pd.read_csv('../../data/rules/dataset_rules.csv')
ds.drop('Unnamed: 0', axis=1, inplace=True)

TESTS = ['../../data/test_rule/rules_left_class_c85.csv',
         '../../data/test_rule/rules_left_class_c90.csv',
         '../../data/test_rule/rules_left_class_c95.csv',
         '../../data/test_rule/pruned_rules_ok.csv',
         '../../data/test_rule/pruned_rules_ok_superset.csv']


def classify(record, csv):
    rules = pd.read_csv(csv)
    ants = rules['Antecedent'].tolist()

    for rule in ants:
        # print 'CONFRONTO CON REGOLA: ' + str(rule)

        rule = literal_eval(rule)
        counter = 0

        for item in rule:
            # print 'CONTROLLO SE ' + str(item) + ' STA NEL RECORD'

            if item not in record:
                # print str(item) + ' NON TROVATO'
                counter = 0
                break
            else:
                # print str(item) + ' TROVATO'
                counter = counter + 1

        if counter == len(rule):
            # print 'TROVATO MATCH CON REGOLA: ' + str(rule)
            return 1
    return 0


target = ds['Left'].map({'Y_L': 1, 'N_L': 0}).tolist()
pred_target = list()

for csv in TESTS:
    print 'CLASSIFICO CON CSV ' + csv
    for record in range(0, 14999):
        row = ds.iloc[record].tolist()
        pred_target.append(classify(row, csv))

    cm = confusion_matrix(target, pred_target)
    acc = metrics.accuracy_score(target, pred_target)
    pre = metrics.precision_score(target, pred_target)
    rec = metrics.recall_score(target, pred_target)

    print 'ACCURACY: ' + str(round(acc, 2))
    print 'PRECISION: ' + str(round(pre, 2))
    print 'RECALL: ' + str(round(rec, 2))

    sns.heatmap(cm, cmap='Spectral', annot=True, cbar=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(fname='../../images/rules/confusion_matrix' +
                csv.split('/')[4].replace('.csv', '') + '.pdf',
                format='pdf', bbox_inches='tight')
    plt.clf()

    pred_target = list()
