import os
import pandas as pd

cars_data1=pd.read_csv('cars.csv',index_col=0)

cars_data2=cars_data1.copy()
cars_data3=cars_data1.copy()
# print(cars_data1.isna().sum())

#If most values of a row are null then delete the row

#In case of numerical values fill the null values with mean or median value

#In case of categorical values fill the numm values with most repeated value
# print(cars_data1.describe()) #Gives statistical information about the data

print(cars_data1['Width'].mean()) #Gives mean of data median function is also available

# cars_data1['Width'].fillna(cars_data1['Width'].mean(),inplace=True)

print(cars_data1['Fuel Type'].value_counts()) #Gives count of all  the unique values in the series
print(cars_data1['Fuel Type'].value_counts().index[0])

cars_data3=cars_data3.apply(lambda x:x.fillna(x.mean()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))
print(cars_data3.isnull().sum())