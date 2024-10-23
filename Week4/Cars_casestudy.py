#Predicting the price of the preowned cars example of regression case study
#Linear regression,Random forest

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Setting the dimensions for the plot
sns.set(rc={'figure.figsize':(11.7,8.27)})

cars_data=pd.read_csv('cars_sampled.csv')
cars=cars_data.copy() #Making a deep copy of the data
# print(cars.info())

# print(cars.describe())
pd.set_option('display.float_format', lambda x: '%.3f' % x) #Changes the foramt of output to float with 3 decimal places 
# print(cars.describe())

# pd.set_option('display.max_columns',500) #Displays maximum possible columns
# print(cars.describe())

#Droping the useless columns(Unwanted columns)

cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=cols,axis=1)
# print(cars.info())
# print(cars_data.info())

#Removing duplicate records

cars.drop_duplicates(keep='first',inplace=True)
# print(cars.info())

#Number of missing values
# print(cars.isnull().sum())

#Variable yearOfRegistration

yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
# print(yearwise_count)

# print(sum(cars['yearOfRegistration']>2018))
# print(sum(cars['yearOfRegistration']<1950))  #We can see that the values less then 1950 and greater than 2018 does not make sense so we have to get rid of them to make a clear sense of this we use scatterplot

# sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)
# plt.show() #From the plot it is clear that the working range of the problem is 1950-2018

#Variable price

price=cars['price'].value_counts().sort_index()
# print(price) #To understand the working range of the problem based on the price we need to plot a histogram

# sns.distplot(cars['price'])
# sns.displot(data=cars, y='price')
# plt.show()

# print(cars.describe())

# sns.boxplot(y=cars['price']) #Plot to check the range of data
# plt.show()

# print(sum(cars['price']>150000)) #We have to get the range by using the plots by trial and error
# print(sum(cars['price']<100)) #We can conclude that the working range in terms of price is 100 to 150000

#Variable powerPS

power_count=cars['powerPS'].value_counts().sort_index()
# print(power_count)

# sns.distplot(cars['price'])
# plt.show()

# print(cars['powerPS'].describe())

# sns.boxplot(y=cars['powerPS'])
# plt.show()

# sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
# plt.show() 

# print(sum(cars['powerPS']>500))
# print(sum(cars['powerPS']<10)) #Hence the working range is 10 to 500

##Final working range for data

cars=cars[(cars.yearOfRegistration<=2018) & (cars.yearOfRegistration>=1950) & (cars.price>=100) & (cars.price<=150000) 
        & (cars.powerPS>=10) & (cars.powerPS<=500)] #6700 records dropped

# print(cars.shape)

#For further simplification we can add a new column age by properly using month and year of registration columns properly
cars['monthOfRegistration']/=12
# print(cars['monthOfRegistration'].sort_values()) #This also get rid of miscalculation where month of registration is 0

cars['Age']=2018-cars['yearOfRegistration']+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
# print(cars['Age'])
# print(cars['Age'].describe())

#As we have incorporated years of registration adn month of registration into age we can drop both of these columns
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)
# print(cars.info())

#Visualizing parameters 

#Age

# sns.distplot(cars['Age'])
# sns.boxplot(y=cars['Age'])
# plt.show()

#Price

# sns.distplot(cars['price'])
# sns.boxplot(y=cars['price'])
# plt.show()

#powerPS

# sns.distplot(cars['powerPS'])
# sns.boxplot(y=cars['powerPS'])
# plt.show()

##Visualizing parameters after reducing working range

#Age v/s Price

# sns.regplot(x='Age',y='price',fit_reg=False,scatter=True,data=cars)
# plt.show() #This plot shows lower the age higher the price

#powerPS v/s price

# sns.regplot(x='powerPS',y='price',fit_reg=False,scatter=True,data=cars)
# plt.show() #Higher the power higher the price


#Seller variable
# print(cars['seller'].value_counts()) #As there is only on row with type 'commercial' it is considered insignificant
# sns.countplot(x='seller',data=cars)
# plt.show()

#OfferType variable 
# print(cars['offerType'].value_counts()) #All are same values hence it is insignificant

#abtest Variable
# print(cars['abtest'].value_counts()) #Equally distributed
# print(pd.crosstab(cars['abtest'],columns='count',normalize=True)) #Shows percentage of distribution
# sns.countplot(x='abtest',data=cars)
# plt.show()

# sns.boxplot(x='abtest',y='price',data=cars)
# plt.show() #From this plot it can be seen that it has almost same distribution as price hence we can declare it insignificant

#Variable vehicle type

# print(cars['vehicleType'].value_counts())
# print(pd.crosstab(cars['vehicleType'],columns='count',normalize=True))
# sns.countplot(x=cars['vehicleType'])
# sns.boxplot(x='vehicleType',y='price',data=cars)
# plt.show()         #This shows that all the 8 types of vehicles affect the price of the car

#Variable gearbox

# print(cars['gearbox'].value_counts())
# print(pd.crosstab(cars['gearbox'],columns='count',normalize=True))
# sns.countplot(x=cars['gearbox'])
# sns.boxplot(x='gearbox',y='price',data=cars)
# plt.show()     #Gearbox affects price

#Variable model

# print(cars['model'].value_counts())
# print(pd.crosstab(cars['model'],columns='count',normalize=True))
# sns.countplot(x=cars['model'])
# sns.boxplot(x='model',y='price',data=cars)
# plt.show() #It is considered in modeling

#Variable kilometer

# print(cars['kilometerec'].value_counts())
# print(pd.crosstab(cars['kilometer'],columns='count',normalize=True))
# sns.countplot(x=cars['kilometer'])
# sns.boxplot(x='kilometer',y='price',data=cars)
# plt.show() #it is considered in modeling as it has a varying range

#Variable fuelType

# print(cars['fuelType'].value_counts())
# print(pd.crosstab(cars['fuelType'],columns='count',normalize=True))
# sns.countplot(x=cars['fuelType'])
# sns.boxplot(x='gearbox',y='price',data=cars)
# plt.show()   #Fuel type affects price

#Variable brand

# print(cars['brand'].value_counts())
# print(pd.crosstab(cars['brand'],columns='count',normalize=True))
# sns.countplot(x=cars['brand'])
# plt.xticks(rotation=90)
# sns.boxplot(x='brand',y='price',data=cars)
# plt.show()  #They are distributed in a wide range hence it is considered

#Variable notRepairedDamage

# yes-cars are damaged and not repaired
# no- cars are damaged and repaired

# print(cars['notRepairedDamage'].value_counts())
# print(pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True))
# sns.countplot(x=cars['notRepairedDamage'])
# plt.xticks(rotation=90)
# sns.boxplot(x='notRepairedDamage',y='price',data=cars)
# plt.show()

#Cars that require repair are under lower price range hence it is reatined

##Removing insignificant variables

col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
# print(cars.info())
cars_copy=cars.copy()

##CORRELATION

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
# print(round(correlation,3))
# print(cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]) #Correlation of price with other numerical variables

#We are going to build 2 typed of models Linear regression and Random forest on 2 distinct sets of data

#OMITTING THE MISSING VALUES

cars_omit=cars.dropna(axis=0)

#Converting categorical variables into dummy variables
cars_omit=pd.get_dummies(cars_omit,drop_first=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #TO build a linear regression model
from sklearn.ensemble import RandomForestRegressor #For the random forest model
from sklearn.metrics import mean_squared_error


#MODEL BUILDING WITH OMITTED DATA

#Separating input and output features

x1=cars_omit.drop(['price'],axis='columns',inplace=False) #This contains everything except price
# print(x1.shape)
y1=cars_omit['price']  #Contains only price
# print(y1.shape)

#plotting the variable price
# prices=pd.DataFrame({'1.Before':y1, '2. After':np.log(y1)})
# prices.hist()
# plt.show() #It is better to consider the natural log of prices

y1=np.log(y1)

#Splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3) 
#By setting the test_size=0.3 the rarion of split of data between train and test becomes 70-30 and random_state is a pre-defined algorithm we can add any number to it
# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


##BUILDING BASELINE MODEL FOR THE OMITTED DATA


#finding the mean for our test data value
base_pred=np.mean(y_test)
# print(base_pred)

#repeating same value till length of test data
base_pred=np.repeat(base_pred,len(y_test))

# Finding the root mean square error it computes the difference between test value and predicted value, squares them and divide them by no. of samples
base_rmse=np.sqrt(mean_squared_error(y_test,base_pred))
# print(base_rmse)


#LINEAR REGRESSION WITH OMITTED DATA


lgr=LinearRegression(fit_intercept=True) #Setting the intercept as true

# Fitting the model on train set
model_lin1=lgr.fit(x_train,y_train)

# predeicting model on test set
cars_prediction_lin1=lgr.predict(x_test)

#Computing MSE and RMSE by passing actually predicted values

lin_mse1=mean_squared_error(y_test,cars_prediction_lin1)
lin_rmse1=np.sqrt(lin_mse1)
# print(lin_rmse1) #The RMSE is dropped by 50%

#R squared value - explains how good our model can explain the variablity in y
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
# print(r2_lin_test1,r2_lin_train1)

#Regression diagnostic -residual plot analysis
residuals1=y_test-cars_prediction_lin1
# sns.regplot(x=cars_prediction_lin1,y=residuals1,scatter=True,fit_reg=False,data=cars)
# plt.show()
# print(residuals1.describe()) 

#As both plot and describe shows that the residuals are near 0 the model is a good model



#RANDOM FOREST MODEL

#Model parameter

rf=RandomForestRegressor(n_estimators=100,max_features='sqrt',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=3) #auto is oudated hence sqrt is used in max_features

# fitting the model
model_rf1=rf.fit(x_train,y_train)

# predicting the model under test set
model_predictions_rf1=rf.predict(x_test)

#Computing MSE and RMSE

rf_mse1=mean_squared_error(y_test,model_predictions_rf1)
rf_rmse1=np.sqrt(rf_mse1)
# print(rf_rmse1)

#R squared values
r2_rf_test1=model_rf1.score(x_test,y_test)
r2_rf_train1=model_rf1.score(x_train,y_train)
# print(r2_rf_test1,r2_rf_train1) #It can be seen that random forest is working better than linear regression


# =================================================================================

# MODEL BUILDING WITH IMPUTED DATA

cars_imputed=cars.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0])) #If the data type is float the missing value will be filled with median otherwise it is replaced by most frequent value
# print(cars_imputed.isnull().sum())

# Converting the categorical coumns into dummy variables
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)
# print(cars_imputed.shape)

# Separating input and output features

x2=cars_imputed.drop(['price'],axis='columns',inplace=False)
y2=cars_imputed['price']

#Plotting variable price

# prices=pd.DataFrame({'1. Before':y2,'2. After':np.log(y2)})
# prices.hist()
# plt.show()

#Transforming y2 to logarithmic values

y2=np.log(y2)

#Splitting the data into train and test
x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3) 
#By setting the test_size=0.3 the rarion of split of data between train and test becomes 70-30 and random_state is a pre-defined algorithm we can add any number to it
# print(x_train1.shape,x_test1.shape,y_train1.shape,y_test1.shape)


##BUILDING BASELINE MODEL FOR THE OMITTED DATA


#finding the mean for our test data value
base_pred1=np.mean(y_test1)
# print(base_pred1)

#repeating same value till length of test data
base_pred1=np.repeat(base_pred1,len(y_test1))

# Finding the root mean square error it computes the difference between test value and predicted value, squares them and divide them by no. of samples
base_rmse1=np.sqrt(mean_squared_error(y_test1,base_pred1))
# print(base_rmse1)



#LINEAR REGRESSION WITH OMITTED DATA


lgr2=LinearRegression(fit_intercept=True) #Setting the intercept as true

# Fitting the model on train set
model_lin2=lgr2.fit(x_train1,y_train1)

# predeicting model on test set
cars_prediction_lin2=lgr2.predict(x_test1)

#Computing MSE and RMSE by passing actually predicted values

lin_mse2=mean_squared_error(y_test1,cars_prediction_lin2)
lin_rmse2=np.sqrt(lin_mse2)
# print(lin_rmse1) #The RMSE is dropped by 50%

#R squared value - explains how good our model can explain the variablity in y
r2_lin_test2=model_lin2.score(x_test1,y_test1)
r2_lin_train2=model_lin2.score(x_train1,y_train1)
# print(r2_lin_test2,r2_lin_train2)

#Regression diagnostic -residual plot analysis
residuals1=y_test-cars_prediction_lin1
# sns.regplot(x=cars_prediction_lin1,y=residuals1,scatter=True,fit_reg=False,data=cars)
# plt.show()
# print(residuals1.describe()) 
#As both plot and describe shows that the residuals are near 0 the model is a good model



#RANDOM FOREST MODEL

#Model parameter

rf2=RandomForestRegressor(n_estimators=100,max_features='sqrt',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=3) #auto is oudated hence sqrt is used in max_features

# fitting the model
model_rf2=rf2.fit(x_train1,y_train1)

# predicting the model under test set
model_predictions_rf1=rf2.predict(x_test1)

#Computing MSE and RMSE

rf_mse2=mean_squared_error(y_test1,model_predictions_rf1)
rf_rmse2=np.sqrt(rf_mse2)
# print(rf_rmse2)

#R squared values
r2_rf_test2=model_rf2.score(x_test1,y_test1)
r2_rf_train2=model_rf2.score(x_train1,y_train1)
print(r2_rf_test1,r2_rf_train1) #It can be seen that random forest is working better than linear regression



# Final output

print("Metrics for models built from data where missing values were omitted")
print("R squared value for train from Linear Regression=  %s"% r2_lin_train1)
print("R squared value for test from Linear Regression=  %s"% r2_lin_test1)
print("R squared value for train from Random Forest=  %s"% r2_rf_train1)
print("R squared value for test from Random Forest=  %s"% r2_rf_test1)
print("Base RMSE of model built from data where missing values were omitted= %s"%base_rmse)
print("RMSE value for test from Linear Regression=  %s"% lin_rmse1)
print("RMSE value for test from Random Forest=  %s"% rf_rmse1)
print("\n\nFor imputed data\n\n")
print("Metrics for models built from data where missing values were imputed")
print("R squared value for train from Linear Regression=  %s"% r2_lin_train2)
print("R squared value for test from Linear Regression=  %s"% r2_lin_test2)
print("R squared value for train from Random Forest=  %s"% r2_rf_train2)
print("R squared value for test from Random Forest=  %s"% r2_rf_test2)
print("Base RMSE of model built from data where missing values were imputed= %s"%base_rmse1)
print("RMSE value for test from Linear Regression=  %s"% lin_rmse2)
print("RMSE value for test from Random Forest=  %s"% rf_rmse2)