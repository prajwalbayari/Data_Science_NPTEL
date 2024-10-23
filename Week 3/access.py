import os
import pandas as pd
import numpy as np

#--------------------------------------------------#

# print(os.getcwd())

# name_data = pd.read_table("trial.txt",delimiter=",")  # Corrected file path with raw string
# print(name_data.head())

# samp=name_data
# print(samp.head()) #Shallow copy

# samp2=name_data.copy(deep=True)
# print(samp2)

# cs1=pd.read_csv("people-100.csv",delimiter=",")

#ACCESSING

# print(cs1.index) #Gives the row label of the data frame
# print(cs1.columns) #Gives the column label of the data frame
# print(cs1.size) #Total number of elements
# print(cs1.shape)
# print(cs1.memory_usage()) #Memory used in bytes
# print(cs1.ndim)
# print(cs1.head(6)) #First 6 rows
# print(cs1.tail(5)) #Last 5 rows
# print(cs1.at[4,'First Name']) #Indexing
# print(cs1.iat[3,3])
# print(cs1.loc[1:10,'First Name']) #Obtain certain values of a column

##DATA TYPES

# print(cs1.dtypes) #Gives data types of all the variables
# print(cs1.select_dtypes(exclude=[object]))
# print(cs1.info())

#UNIQUE FUNCTION

# print(np.unique(cs1['First Name'])) #Gives all the unique values of the column

# cs1['Index']=cs1['First Name'].astype('object') #Changes data type of the column
# print(cs1.dtypes)

# print(cs1['First Name'].nbytes)
# print(cs1["First Name"].astype('category').nbytes) #Gives the total bytes consumed by elements of the column

# cs1['Last Name'].replace('Terrell','Jackson',inplace=True)
# print(cs1["Last Name"].head(5)) #Replace values of columns

# np.where()

# print(cs1.isnull().sum()) #Gives the number of null values in each colums

#-----------------------------------------------------------------------------------------------------------------------------#

#  EXPLORATORY DATA ANALYSIS

cs1=pd.read_csv("customers-1000.csv")

copied=cs1.copy(deep=False)

#FREQUENCY TABLE

x=pd.crosstab(index=copied['Country'],columns='count',dropna=True)  
# print(x)

#TWO WAY TABLE

x=pd.crosstab(index=cs1['Country'],columns=cs1['City'],dropna=True) 
# print(x)

## Joint probability
x=pd.crosstab(index=cs1['Country'],columns=cs1['City'],dropna=True,normalize=True) 
# print(x) #Gives proabability of cross event

## Marginprobabilty
x=pd.crosstab(index=cs1['Country'],columns=cs1['City'],dropna=True,normalize=True,margins=True)
# print(x) #Gives row sum and column of probability

##Conditional probability 
x=pd.crosstab(index=cs1['Country'],columns=cs1['City'],dropna=True,normalize='columns',margins=True)
# print(x)


#CORRELATION

cs1=pd.read_csv('cars.csv')
numerical=cs1.select_dtypes(exclude='object')

# print(numerical.shape)
matrix=numerical.corr()
print(matrix)

#-------------------------------------------------------------------------------------------------------------------------------#


# cs1=pd.read_csv('people-100.csv')
# print(cs1['Sex'].value_counts())