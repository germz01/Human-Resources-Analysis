import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
FIG = plt.figure(figsize=(6, 5))


def plot_cat_and_ord(col):
    lef = np.arange(len(DF[col].unique()))
    plt.bar(x=lef, height=STAYED[col].value_counts().tolist(),
            width=0.35, label='Employees Who Stayed')
    plt.bar(x=lef+0.35, height=LEFT[col].value_counts().tolist(),
            width=0.35, label='Employees Who left')
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
    pass


def plot_continous(col):
    pass

if __name__ == '__main__':
    to_plot = int(raw_input('What column do you want to plot?\n1: ' +
                            'Satisfaction_Level\n2: Last_Evaluation\n' +
                            '3: Number_Project\n4: Average_Montly_Hours\n' +
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
            plot_discrete(COLUMNS[to_plot][0])
        else:
            plot_continous(COLUMNS[to_plot][0])

        plt.savefig('../images/' + COLUMNS[to_plot][0].lower() + '.pdf')
