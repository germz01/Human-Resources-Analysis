import pandas as pd
import csv

DS = pd.read_csv(filepath_or_buffer='/Users/gianmarco/Documents/Università/' +
                 'Magistrale/Data Mining/Human-Resources-Analysis/' +
                 'HR_comma_sep.csv',
                 sep=',',
                 header=0,
                 names=['Satisfaction Level', 'Last Evaluation',
                        'Number Project', 'Average Montly Hours',
                        'Time Spend Company', 'Work Accident', 'Left',
                        'Promotion Last 5 Years', 'Sales', 'Salary'])

descr_stat = DS.describe(include='all')
csvfile = open('descriptive_statistics.csv', 'wb')
writer = csv.DictWriter(csvfile, ['Field', 'Min', 'Max', 'Mean',
                                  'Standard Deviation', '25% Percentile',
                                  '50% Percentile', '75% Percentile', 'Top',
                                  'Frequence'])
writer.writeheader()

for c in DS.columns:
    current = descr_stat[c]
    if c not in ['Sales', 'Salary']:
        writer.writerow({'Field': c, 'Min': current['min'],
                        'Max': current['max'], 'Mean': current['mean'],
                         'Standard Deviation': current['std'],
                         '25% Percentile': current['25%'],
                         '50% Percentile': current['50%'],
                         '75% Percentile': current['75%'],
                         'Top': '', 'Frequence': ''})
    else:
        writer.writerow({'Field': c, 'Min': '', 'Max': '', 'Mean': '',
                        'Standard Deviation': '', '25% Percentile': '',
                         '50% Percentile': '', '75% Percentile': '',
                         'Top': current['top'], 'Frequence': current['freq']})

csvfile.close()

# for c in DS.columns:
#     csv_file = open(c.lower() + '.csv', 'wb')
#     writer = csv.writer(csv_file, delimiter=',')
#     current = descr_stat[c]

#     writer.writerow(['Min', 'Max', 'Mean', 'Standard Deviation',
#                     '25% Percentile', '50% Percentile', '75% Percentile'])
#     writer.writerow([current['min'], current['max'], current['mean'],
#                     current['std'], current['25%'], current['50%'],
#                     current['75%']])
#     csv_file.close()
