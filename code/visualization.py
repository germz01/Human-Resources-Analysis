import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

sns.set()
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y label
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

DF = pd.read_csv('../HR_comma_sep.csv',
                 names=['Satisfaction_Level', 'Last_Evaluation',
                        'Number_Project', 'Average_Montly_Hours',
                        'Time_Spend_Company', 'Work_Accident', 'Left',
                        'Promotion_Last_5_Years', 'Sales', 'Salary'],
                 header=0)

LEFT, STAYED = DF[DF.Left == 1], DF[DF.Left == 0]
COLUMNS = {1: ['Satisfaction_Level', 'continous'],
           2: ['Last_Evaluation', 'continous'],
           3: ['Number_Project', 'discrete'],
           4: ['Average_Montly_Hours', 'discrete'],
           5: ['Time_Spend_Company', 'discrete'],
           6: ['Work_Accident', 'discrete'],
           7: ['Promotion_Last_5_Years', 'discrete'],
           8: ['Sales', 'categorical'],
           9: ['Salary', 'ordinal']}

plt.style.use('ggplot')
FIG = plt.figure(figsize=(8, 5))


def fill_values(df, col, type):
    if type is 'discrete':
        ticks = DF[col].value_counts().to_dict().keys()
    else:
        ticks = DF[col].value_counts().keys()

    dic = df[col].value_counts()

    return [dic[i] if i in dic.keys() else 0 for i in ticks]


def plot(col, type):
    d = DF[col].value_counts()

    if type is 'continous' or col is 'Average_Montly_Hours':
        ideal_bins = int(math.ceil(
                                math.log(float(len(DF[col].tolist())), 2)) + 1)
        plt.hist(x=[LEFT[col].tolist(), DF[col].tolist()],
                 bins=ideal_bins, stacked=True,
                 color=['tomato', 'steelblue'], label=['Left', 'Stayed'])
    elif type is 'discrete':
        plt.bar(x=np.arange(len(d.keys())), height=d.to_dict().values(),
                color='steelblue', label='Stayed')
        plt.bar(x=np.arange(len(d.keys())),
                height=fill_values(LEFT, col, type),
                color='tomato', label='Left')
        plt.xticks(np.arange(len(d.keys())), d.to_dict().keys())
    else:
        plt.bar(np.arange(len(d.keys())), height=d.get_values(),
                color='steelblue', label='Stayed')
        plt.bar(np.arange(len(d.keys())), height=fill_values(LEFT, col, type),
                label='Left', color='tomato')
        if col == 'Sales':
            plt.xticks(np.arange(len(d.keys())), d.keys(), rotation='vertical')
        else:
            plt.xticks(np.arange(len(d.keys())), d.keys())

    if col == 'Sales':
        plt.xlabel('Departments')
        plt.title('Department variable distribution')
    else:
        plt.xlabel(col.replace('_', ' '))
        plt.title(col.replace('_', ' ') + ' variable distribution')

    plt.ylabel('Employees')
    plt.legend()


if __name__ == '__main__':
    while True:
        to_plot = int(raw_input('What column do you want to plot?\n1: ' +
                                'Satisfaction_Level\n2: Last_Evaluation\n' +
                                '3: Number_Project\n' +
                                '4: Average_Montly_Hours\n' +
                                '5: Time_Spend_Company\n6: Work_Accident\n' +
                                '7: Promotion_Last_5_Years\n8: Sales\n' +
                                '9: Salary\n10: All\n\n'))

        if to_plot not in range(1, 11):
            print 'ERROR'
        elif to_plot != 10:
            plot(COLUMNS[to_plot][0], COLUMNS[to_plot][1])
            plt.savefig(fname='../images/newplottedvariables/' +
                        COLUMNS[to_plot][0].lower() + '.pdf',
                        format='pdf', bbox_inches='tight')
        else:
            for i in xrange(1, 10):
                print 'PLOTTING ' + COLUMNS[i][0]
                plot(COLUMNS[i][0], COLUMNS[i][1])
                plt.savefig(fname='../images/newplottedvariables/' +
                            COLUMNS[i][0].lower() + '.pdf',
                            format='pdf', bbox_inches='tight')
                plt.clf()

        if raw_input('Print more?(Yes/No) ') in ['No', 'N', 'no', 'n']:
            plt.close()
            break

        plt.clf()
