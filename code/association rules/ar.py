import csv
import math
import pandas as pd
from fim import apriori


def sturges(n=14999):
    k = math.ceil(math.log(n, 2)) + 1

    return int(k)


print 'IMPORTING THE DATA SET'

DS = pd.read_csv(filepath_or_buffer='../../data/df_formatted.csv')

print 'FORMATTING THE DATA SET'

DS['Salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace=True)
DS['Department'].replace(['sales', 'technical', 'support', 'IT',
                          'product_mng', 'marketing', 'RandD', 'accounting',
                          'hr', 'management'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         inplace=True)
DS['SL_100'] = DS['Satisfaction_Level']*100
DS['LE_100'] = DS['Last_Evaluation']*100
DS['AMHGroup'] = pd.cut(DS['Average_Montly_Hours'], bins=sturges(),
                        right=False, labels=range(0, 15))
DS['LEGroup'] = pd.cut(DS['LE_100'], bins=sturges(), right=False,
                       labels=range(0, 15))
DS['SLGroup'] = pd.cut(DS['SL_100'], bins=sturges(), right=False,
                       labels=range(0, 15))
DS['Work_Accident'] = DS['Work_Accident'].map({1: 'Y',
                                              0: 'N'}).astype(str) + '_WA'
DS['Left'] = DS['Left'].map({1: 'Y', 0: 'N'}).astype(str) + '_L'
DS['Promotion_Last_5_Years'] = DS['Promotion_Last_5_Years']. \
                               map({1: 'Y', 0: 'N'}).astype(str) + '_P'
DS['Department'] = DS['Department'].astype(str) + '_D'
DS['Number_Projects'] = DS['Number_Projects'].astype(str) + '_NP'
DS['SLGroup'] = DS['SLGroup'].astype(str) + '_SL'
DS['LEGroup'] = DS['LEGroup'].astype(str) + '_LE'
DS['Time_Spend_Company'] = DS['Time_Spend_Company'].astype(str) + '_T'
DS['AMHGroup'] = DS['AMHGroup'].astype(str) + '_H'
DS['Salary'] = DS['Salary'].astype(str) + '_S'

DS.drop(['Satisfaction_Level', 'Last_Evaluation', 'Average_Montly_Hours',
        'SL_100', 'LE_100'], axis=1, inplace=True)

print 'APPLYING APRIORI ALGORITHM'

records = DS.to_records(index=False)

for target in ['s', 'c', 'm']:
    itemsets = apriori(records, supp=20, zmin=2, target=target, report='s')

    print 'SAVING FREQUENT ITEMSETS FOR TARGET ' + target + ' IN CSV FILE ' \
          '"../../data/frequent_itemsets_' + target + '.csv"'

    with open('../../data/frequent_itemsets_' + target + '.csv', 'wb') as f:
        fieldnames = ['ITEMSET', 'SUPPORT']
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()
        for record in itemsets:
            support = str(round(float(record[1]), 2))
            csv_writer.writerow({'ITEMSET': record[0], 'SUPPORT': support})

print 'MINING ASSOCIATION RULES'

rules = apriori(records, supp=20, zmin=2, target='r', conf=80, report='cl')

print 'SAVING ASSOCIATION RULES IN CSV FILE "../../data/association_rules.csv"'

with open('../../data/association_rules.csv', 'wb') as f:
    fieldnames = ['RULE', 'CONFIDENCE', 'LIFT']
    csv_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
    csv_writer.writeheader()
    for record in rules:
        csv_writer.writerow({'RULE': record[0] + ' -> ' + str(record[1]),
                            'CONFIDENCE': record[2], 'LIFT': record[3]})
