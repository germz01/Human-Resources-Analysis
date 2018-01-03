import pandas as pd
import csv

DS = pd.read_csv(filepath_or_buffer='../../data/df_formatted.csv')
left = DS[DS.Left == 1]
stayed = DS[DS.Left == 0]
columns = DS.columns.tolist()

with open('../../data/df_formatted_ordered.csv', 'w') as csvfile:
    fieldnames = columns
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    i = 0
    for row in left.as_matrix():
        lst = row.tolist()
        writer.writerow({columns[0]: lst[0],
                         columns[1]: lst[1],
                         columns[2]: lst[2],
                         columns[3]: lst[3],
                         columns[4]: lst[4],
                         columns[5]: lst[5],
                         columns[6]: lst[6],
                         columns[7]: lst[7],
                         columns[8]: lst[8],
                         columns[9]: lst[9]})
    for row in stayed.as_matrix():
        lst = row.tolist()
        writer.writerow({columns[0]: lst[0],
                         columns[1]: lst[1],
                         columns[2]: lst[2],
                         columns[3]: lst[3],
                         columns[4]: lst[4],
                         columns[5]: lst[5],
                         columns[6]: lst[6],
                         columns[7]: lst[7],
                         columns[8]: lst[8],
                         columns[9]: lst[9]})
