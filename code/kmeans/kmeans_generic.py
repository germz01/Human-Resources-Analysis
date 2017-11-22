import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

DS = DS = pd.read_csv('../HR_comma_sep.csv',
                      names=['Satisfaction_Level', 'Last_Evaluation',
                             'Number_Project', 'Average_Montly_Hours',
                             'Time_Spend_Company', 'Work_Accident', 'Left',
                             'Promotion_Last_5_Years', 'Sales', 'Salary'],
                      header=0)

sns.set()

if __name__ == '__main__':
    pass
