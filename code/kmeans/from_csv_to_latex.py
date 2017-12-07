import pandas as pd

columns = ['Average_Montly_Hours', 'Last_Evaluation', 'Number_Project',
           'Satisfaction_Level', 'Time_Spend_Company', 'Left']
df = pd.read_csv('../../data/kmeans_distribution.csv')

print df.columns

for i in xrange(4, 7):
    for col in columns:
        df.drop(labels=[col + '.' + str(i)], axis=1, inplace=True)

file = open('./kmeans_distribution.tex', 'wb')
file.write('\\begin{table}[H]\n\t\centering\n\t')

for col in columns:
    l = list()

    file.write('\\begin{subtable}{0.4\\textwidth}\n\t\t'
               '\\resizebox{\\textwidth}{!}{\n\t\t'
               '\\begin{tabular}{| c | c | c | c | c | c |}\n\t\t'
               '\\hline\n\t\t'
               '{} & \multicolumn{5}{c |}{' + col + '} \\\\\n\t\t'
               '\\hline\n\t\t'
               '{} & count & mean & std & min & max \\\\\n\t\t'
               'Cluster & & & & &  \\\\\n\t\t'
               '\\hline\n\t\t')

    for i in range(0, 4):
        l.append([df['Unnamed: 0'][i + 2]])
        l[i].append(df[col][i + 2])
        for j in [1, 2, 3, 7]:
            l[i].append(df[col + '.' + str(j)][i + 2])

    for line in l:
        for x in range(0, len(line)):
            if x != len(line) - 1:
                file.write(line[x] + ' & ')
            else:
                file.write(line[x] + ' \\\\\n\t\t')

    file.write('\\hline\n\t\t'
               '\end{tabular}\n\t}\n\t\\caption{}\n\t\\label{tab:}\n\t'
               '\end{subtable}\n\t')
file.write('\end{table}')
file.close()
