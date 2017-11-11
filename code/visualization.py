import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

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


def plot_cat_and_ord(col):
    """
        This function plots the categorical and ordinal columns of the data
        set. Each column is plotted drawing the datas for the employees who
        left and the employees who stayed on the same graphic.
    """

    lef = np.arange(len(DF[col].unique()))
    plt.bar(x=lef, height=STAYED[col].value_counts().tolist(),
            width=0.35, label='Employees Who Stayed', color='tomato')
    plt.bar(x=lef+0.35, height=LEFT[col].value_counts().tolist(),
            width=0.35, label='Employees Who left', color='steelblue')
    if col == 'Sales':
        plt.xticks(np.arange(len(DF[col].unique())) + (0.35/2),
                   DF[col].unique(), rotation='vertical', fontsize='3')
    else:
        plt.xticks(np.arange(len(DF[col].unique())) + (0.35/2),
                   DF[col].unique())
    plt.ylabel('Employees')
    plt.legend()
    plt.title(col.replace('_', ' ') + ' per Employee')


def plot_discrete(col):
    """
        This function plots the discrete columns of the data set. Each column
        is plotted drawing the datas for the employees who left and the
        employees who stayed on the same graphic.
    """

    min, max = DF[col].describe()['min'], DF[col].describe()['max']
    lef = np.arange(min, max + 1)
    value_dict = STAYED[col].value_counts().to_dict()
    h = []

    for i in xrange(int(min), int(max + 1)):
        if value_dict.has_key(i):
            h.append(value_dict[i])
        else:
            h.append(0)

    plt.bar(x=lef, height=h, width=0.35, label='Employees Who Stayed',
            color='tomato')

    value_dict = LEFT[col].value_counts().to_dict()
    h = []

    for i in xrange(int(min), int(max + 1)):
        if value_dict.has_key(i):
            h.append(value_dict[i])
        else:
            h.append(0)

    plt.bar(x=lef + 0.35, height=h, width=0.35, label='Employees Who left',
            color='steelblue')

    plt.xticks(np.arange(min, max + 1) + (0.35/2),
               np.arange(min, max + 1))
    plt.ylabel('Employees')
    plt.legend()
    plt.title(col.replace('_', ' ') + ' per Employee')


def plot_continous(col):
    ideal_bins = int(math.ceil(math.log(float(len(DF[col].tolist())), 2)) + 1)

    plt.hist(x=[LEFT[col].tolist(), STAYED[col].tolist()], bins=ideal_bins,
             stacked=True, color=['tomato', 'steelblue'],
             label=['Left', 'Stayed'])
    plt.xlabel(col.replace('_', ' '))
    plt.ylabel('Employees')
    plt.legend()

    if col == 'Average_Montly_Hours':
        plt.xticks(np.arange(50, 350, 50))
    elif col == 'Last_Evaluation':
        plt.xticks(np.linspace(0.25, 1.0, 4))
    else:
        plt.xticks(np.linspace(0.0, 1.0, 5))

    plt.title(col.replace('_', ' ') + ' per Employee')

if __name__ == '__main__':
    while True:
        to_plot = int(raw_input('What column do you want to plot?\n1: ' +
                                'Satisfaction_Level\n2: Last_Evaluation\n' +
                                '3: Number_Project\n' +
                                '4: Average_Montly_Hours\n' +
                                '5: Time_Spend_Company\n6: Work_Accident\n' +
                                '7: Promotion_Last_5_Years\n8: Sales\n' +
                                '9: Salary\n\n'))

        if to_plot not in range(1, 10):
            print 'ERROR'
        else:
            type = COLUMNS[to_plot][1]

            if type == 'ordinal' or type == 'categorical':
                plot_cat_and_ord(COLUMNS[to_plot][0])
            elif type == 'discrete':
                if COLUMNS[to_plot][0] == 'Average_Montly_Hours':
                    plot_continous(COLUMNS[to_plot][0])
                else:
                    plot_discrete(COLUMNS[to_plot][0])
            else:
                plot_continous(COLUMNS[to_plot][0])

            plt.savefig(fname='../images/' + COLUMNS[to_plot][0].lower() +
                        '.pdf',
                        format='pdf', bbox_inches='tight')

        if raw_input('Print more?(Yes/No) ') in ['No', 'N', 'no', 'n']:
            plt.close()
            break

        plt.clf()
