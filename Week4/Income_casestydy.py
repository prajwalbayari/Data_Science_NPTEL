#This is a classification case study
#Logistic Regression,Random forest and KNN

import os 
import pandas as pd #To work with dataframes
import numpy as np #To perform numerical operations
import seaborn as sns #To visualize data
import matplotlib.pyplot as mp #To partition data
from sklearn.model_selection import train_test_split #To partition data
from sklearn.linear_model import LogisticRegression #importing library from logistic regression
from sklearn.metrics import accuracy_score,confusion_matrix #Importing performing metrics accuracy score,confusion matrix
from sklearn.preprocessing import StandardScaler

data_income=pd.read_csv('income.csv')
data=data_income.copy()

#Exploratory data analysis
# 1.getting to know the data 
# 2.Data preprocessing (Missing values)
# 3.Cross tables and data visualizations

# 1.getting to know the data 

# print(data.info()) #Check variables data type

# print(data.isnull().sum())

#Summary of the data

sum_num=data.describe() #Summary of numerical variables
# print(sum_num)
sum_cat=data.describe(include="O") #Summary of categorical variable
# print(sum_cat)

#Frequency of each category

# print(data['JobType'].value_counts())
# print(data['occupation'].value_counts()) #Both have  ? as null value which is not identified by the isNull function

# Checking for  unique classes in each category

# print(np.unique(data['JobType']))
# print(np.unique(data['occupation']))

#read the file again and replace ? with na value

data=pd.read_csv('income.csv',na_values=[' ?'])

# print(data.isnull().sum())

missing =data[data.isnull().any(axis=1)] #axis=1 to consider atleast one value is missing 

# print(missing)

##Points to be noted
#1.Missing values in jobtype=1809
#2.Missing values in occupation=1816
#3.There are 1809 rows where 2 columns have nan values
#4.7 have occupation unfilled because never had a job

data2=data.dropna(axis=0)
# print(data.size-data2.size)

#Relatiobnship between independent values

numeric=data2.select_dtypes(exclude='object')
correlation=numeric.corr()
# print(correlation)


#CROSS TABLES AND DATA VISUALIZATIONS

#Extracting the column names

# print(data2.columns)

gender=pd.crosstab(index=data2['gender'],columns='count',normalize=True)  #Gender proportion table
# print(gender)

gender_salary=pd.crosstab(index=data2['gender'],columns=data2['SalStat'],margins=True,normalize='index') #Relationship between gender and salary
# print(gender_salary)

# SalStat=sns.countplot(x=data2['SalStat'])  #A graph to deduce the distribution of salary above and below 50k
# mp.show()

# sns.displot(data2['age'],bins=10,kde=False) #A plot to deduce the distribution of salary based on age
# mp.show()

# sns.boxplot(x='SalStat',y='age',data=data2) #Shows the variation of salary along with age
# mp.show()

# print(data2.groupby('SalStat')['age'].median())

# sns.countplot(y='JobType',data=data2,hue='SalStat')          #Job type v/s Salary graph
# mp.ylabel('Job types')
# mp.show()

# list=pd.crosstab(data2['JobType'],data2['SalStat'],margins=True,normalize='index')*100  #Crosstable for jobtype v/s salary
# list=list.map(lambda x: f'{x:.2f}')
# print(list)

# sns.countplot(y='EdType',data=data2,hue='SalStat')          #Education v/s Salary graph
# mp.show()

# list=pd.crosstab(data2['EdType'],data2['SalStat'],margins=True,normalize='index')*100  #Crosstable for Education v/s salary
# list=list.map(lambda x: f'{x:.1f}')
# print(list)

# sns.countplot(y='occupation',data=data2,hue='SalStat')          #occupation v/s Salary graph
# mp.show()

# list=pd.crosstab(data2['occupation'],data2['SalStat'],margins=True,normalize='index')*100  #Crosstable for Occupation v/s salary
# list=list.map(lambda x: f'{x:.1f}')
# print(list)

# sns.displot(x=data2['capitalgain'],bins=10,kde=False)  #A graph to deduce the capital gain
# mp.show()

# sns.displot(x=data2['capitalloss'],bins=10,kde=False)  #A graph to deduce the capital loss
# mp.show()

# sns.boxplot(y='hoursperweek',x='SalStat',data=data2) #Shows the variation of salary along with age
# mp.xlabel('Salary Status')
# mp.ylabel('Hopurs spent per week')
# mp.show()

##LOGISTIC REGRESSION MODEL

#Reindexing salary status names to 0 and 1

# data2.loc[:,'SalStat'] = data2['SalStat'].map({" less than or equal to 50,000":0, " greater than 50,000":1})
data2=data2.copy()
data2['SalStat'] = data2['SalStat'].map({" less than or equal to 50,000": 0, " greater than 50,000": 1})

# print(data2['SalStat'])

new_data=pd.get_dummies(data2,drop_first=True)
# print(new_data)

#Storing the column names
columns_list=list(new_data.columns)
# print(columns_list)

#Separating input names from data
features=list(set(columns_list)-set(['SalStat']))
# print(features)

#Storing the output values in y
y=new_data['SalStat'].values
# print(y)

#Storing the values from input features
x=new_data[features].values
# print(x)

#Splitting the data into train and test set
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

scaler=StandardScaler() #Ensures that features are approximately scaled
train_x=scaler.fit_transform(train_x)
test_x=scaler.fit_transform(test_x)

#Make an intsance of the model
logistic=LogisticRegression(max_iter=5000,solver='liblinear') #Setting the number of iteration

#Fitting the values for x and y
logistic.fit(train_x,train_y)
# print(logistic.intercept_)

#Prediction from test data
prediction=logistic.predict(test_x)
# print(prediction)

#Confusion matrix
confusionmatrix=confusion_matrix(test_y,prediction)
# print(confusionmatrix)

#Calculating accuracy 
accuracyscore=accuracy_score(test_y,prediction)
# print(accuracyscore)

#Printing missclassified values from prediction
# print("Misclassified values: %d"%(test_y!=prediction).sum())

# print(data2['SalStat'])

#REMOVING INSIGNIFICANT VARIABLES

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(new_data,drop_first=True)

#Storing column names
columns_list=list(new_data.columns)
# print(columns_list)

#Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
# print(features)

#Storing output values in y
y=new_data['SalStat'].values
# print(y)

#Storing the values from input features
x=new_data[features].values
# print(x)

#Splitting the data into train and test set
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)

scaler=StandardScaler()  #Ensures that features are approximately scaled
train_x=scaler.fit_transform(train_x)
test_x=scaler.fit_transform(test_x)

#Make an intsance of the model
logistic=LogisticRegression(max_iter=1000,solver='liblinear') #Setting the number of iteration

#Fitting the values for x and y
logistic.fit(train_x,train_y)
# print(logistic.intercept_)

#Prediction from test data
prediction=logistic.predict(test_x)
# print(prediction)

#Calculating accuracy 
accuracyscore=accuracy_score(test_y,prediction)
print(accuracyscore)

#Printing missclassified values from prediction
print("Misclassified values: %d"%(test_y!=prediction).sum())


#KNN
# ==========================================================================================================================================
from sklearn.neighbors import KNeighborsClassifier #imorting the KNN library
import matplotlib.pyplot as plt #Import for plotting

#Storing the K nearest neighbour classifier
knn=KNeighborsClassifier(n_neighbors=5)

#Fitting the values for X and Y
knn.fit(train_x,train_y)

#Predicting the test values with the model
prediction=knn.predict(test_x)
# print(prediction)

#Performance metric check
confusionmatrix=confusion_matrix(test_y,prediction)
# print("\t","Predicted values")
# print("Original values","\n",confusionmatrix)

#Calculating the accuracy
accuracyscore=accuracy_score(test_y,prediction)
# print(accuracyscore)

# print("Missclassified samples :%d" %(test_y!=prediction).sum())

#Effect of K value on classifier

# Missclassified_sample=[]
# for i in range(1,20):
#     knn=KNeighborsClassifier(n_neighbors=i)
#     knn.fit(train_x,train_y)
#     pred_i=knn.predict(test_x)
#     Missclassified_sample.append((test_y!=prediction).sum())

# print(Missclassified_sample)