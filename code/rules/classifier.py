import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.metrics import confusion_matrix

sns.set()
sns.set_style("whitegrid")
FIG = plt.figure(figsize=(8, 5))

ds = pd.read_csv('../../data/rules/dataset_rules.csv')
ds.drop('Unnamed: 0', axis=1, inplace=True)


def classify(record):
    rules = pd.read_csv('~/Desktop/rules_left_class_c85.csv')
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

for record in range(0, 14999):
    row = ds.iloc[record].tolist()
    pred_target.append(classify(row))

cm = confusion_matrix(target, pred_target)

sns.heatmap(cm, cmap='Spectral', cbar=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.savefig(fname='../../images/rules/confusion_matrix.pdf', format='pdf',
            bbox_inches='tight')
