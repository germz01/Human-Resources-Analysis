import pandas as pd


df = pd.read_csv('../HR_comma_sep.csv',
                 names=['Satisfaction_Level', 'Last_Evaluation',
                        'Number_Project', 'Average_Montly_Hours',
                        'Time_Spend_Company', 'Work_Accident', 'Left',
                        'Promotion_Last_5_Years', 'Sales', 'Salary'],
                 header=0)


df_shape=df.shape

## choose an ordering!
df_formatted = df[['Satisfaction_Level',
                   'Last_Evaluation',
                   'Average_Montly_Hours',
                   'Time_Spend_Company',
                   'Number_Project',
                   'Work_Accident',
                   'Promotion_Last_5_Years',
                   'Sales',
                   'Salary',
                   'Left'
]]

df_formatted.rename(columns={'Sales':'Department',
                             'Number_Project':'Number_Projects'
}, inplace=True)

## check and write
df_formatted_shape=df_formatted.shape

if df_shape==df_formatted_shape:
    df_formatted.to_csv('../data/df_formatted.csv',index=False)
    

# test
#df_test = pd.read_csv('../data/df_formatted.csv',header=0)
#df_test.head()

