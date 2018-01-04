import csv
import numpy as np
import pandas as pd
from fim import apriori


print 'IMPORTING THE DATA SET'

DS = pd.read_csv(filepath_or_buffer='../../data/df_formatted.csv')

print 'FORMATTING THE DATA SET'

# salario va già bene con 'low'....
#DS['Salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace=True)

DS['Department'].replace(['sales', 'technical', 'support', 'IT',
                          'product_mng', 'marketing', 'RandD', 'accounting',
                          'hr', 'management'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         inplace=True)

DS['SL_100'] = DS['Satisfaction_Level']*100
DS['LE_100'] = DS['Last_Evaluation']*100
###########################################################
#cutting delle variabili numeriche
DS['AMHGroup'] = pd.cut(DS['Average_Montly_Hours'], bins=[0, 200, 311],
                        right=False, labels=['standard', 'intensive'])

DS['LEGroup'] = pd.cut(DS['SL_100'], bins=[0, 45, 57, 77, 101], right=False,                       labels=['insufficient', 'sufficient', 'good',
                       'very good'])
DS['SLGroup'] = pd.cut(DS['SL_100'], bins=[0, 33, 66, 101], right=False,
                       labels=['low', 'medium', 'high'])
###########################################################

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

#check visivo for Nan
DS.LEGroup.value_counts()
DS.SLGroup.value_counts()
DS.AMHGroup.value_counts()


DS.to_csv('../../data/dataset_rules.csv')

records = DS.to_records(index=False)

###########################################################
# a priori for frequent_itemsets_
#supp = int(raw_input('SUPPORT: '))
supp=20


for target in ['s', 'c', 'm']:
    itemsets = apriori(records, supp=supp, zmin=2, target=target, report='s',mode='o')

    print 'EXTRACTED ' + str(len(itemsets)) + ' FREQUENT ITEMSETS'
    print 'SAVING FREQUENT ITEMSETS WITH SUPPORT ' + str(supp) + ' AND ' \
          'TARGET ' + target + ' IN CSV FILE "../../data/frequent_itemsets_' \
          + str(supp) + '_' + target + '.csv"'

    fname = '../../data/frequent_itemsets_' + str(supp) + '_' + target + '.csv'

    with open(fname, 'wb') as f:
        fieldnames = ['ITEMSET', 'SUPPORT']
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()
        for record in itemsets:
            support = str(round(float(record[1]), 2))
            csv_writer.writerow({'ITEMSET': record[0], 'SUPPORT': support})

###########################################################

# inizio analisi Ste: estrazione delle regole

def extract_rules(supp,conf):

    report_dict = {'s':'support','c':'confidence','l':'lift'}
    report = "".join(report_dict.keys())
    #report_list = report.split()
    for target in ['r']:
        out = apriori(records, supp=supp, conf=conf,
                      zmin=2,target=target, report=report,
                      mode='o')

    ###########################################################
    ## scrittura su file        
    fieldnames = ['Consequent', 'Antecedent',
                  'Supp','Conf','Lift']

    fname = '../../data/rules/rules_supp{}_conf{}.csv'.format(supp,conf)

    with open(fname, 'wb') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()

        for record in out:
            ## costruisco il dizionario per writerow
            csv_dict = {fieldnames[0] : record[0]}
            for i,field in enumerate(fieldnames):
                if type(record[i]) is float:
                    csv_dict.update({field :round(record[i],2)})
                else:
                    csv_dict.update({field : record[i]})

            csv_writer.writerow(csv_dict)
        print 'scrivo su disco'

###########################################################    


supp=20
conf_range=np.arange(50,105,5)

for conf in conf_range:
    extract_rules(supp,conf)


supp=10
conf_range=np.arange(50,105,5)

for conf in conf_range:
    extract_rules(supp,conf)
    

supp = 5
conf_range=np.arange(50,105,5)
for conf in conf_range:
    extract_rules(supp,conf)




