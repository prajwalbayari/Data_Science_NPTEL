import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cars_data1=pd.read_csv('cars.csv',index_col=0)
# print(cars_data1.head(5))

##MATPLOTLIB

#Scatter plot

# plt.scatter(cars_data1['City mileage'],cars_data1['Highway mileage'],c='red')
# plt.title('Cars mileage')
# plt.xlabel('City mileage in mph')
# plt.ylabel('Highway mileage in mph')
# plt.show()

#Histogram

# plt.hist(cars_data1['Width'],color='green',edgecolor='white',bins=5)
# plt.xlabel('Width in cm')
# plt.title('Width of various cars')
# plt.ylabel('Frequency')
# plt.show()

#Bar plot

# counts=[97,120,12]
# fuel=('Petrol','Diesel','CNG')
# index=np.arange(len(fuel))

# plt.bar(index,counts,color=['red','blue','cyan'])
# plt.title('Bar plot of fuel types')
# plt.xlabel('Fuel types')
# plt.ylabel('Frequency',rotation=90)
# plt.xticks(index,fuel)
# plt.show()


##SEABORN

# pop = pd.read_csv('2.csv', index_col=0, encoding='ISO-8859-1')

#Scatter plot
# sns.set(style="darkgrid")
# sns.relplot(x=cars_data1['Highway mileage'],y=cars_data1['City mileage'],marker='*')
# plt.show()

#LMplot

# sns.set(style='darkgrid')
# sns.lmplot(x='Highway mileage',y='City mileage',data=cars_data1,fit_reg=False,hue='Classification',legend=True,palette='Set1')
# plt.show()

#Histogram

# sns.set(style='darkgrid')
# sns.histplot(cars_data1['Length'],kde=False,bins=20) #distplot will be remoevd in new version
# plt.show()

#Barplot

# sns.set(style='darkgrid')
# sns.countplot(x='Driveline',data=cars_data1)
# plt.show()

#Grouped bar plot

# sns.set(style='darkgrid')
# sns.countplot(x='Driveline',data=cars_data1,hue="Number of Forward Gears")
# plt.show()

#Box and whiskers plot

#1.Numerical

# sns.set(style='darkgrid')
# sns.boxplot(y=cars_data1['Width'])
# plt.show()

#2.Numerical and categorical

# print(np.unique(cars_data1['Fuel Type']))

# sns.set(style='darkgrid')
# sns.boxplot(x=cars_data1["Fuel Type"],y=cars_data1['Width'])
# plt.show()

#Grouped box and whiskers plot

# sns.set(style='darkgrid')
# sns.boxplot(x=cars_data1["Fuel Type"],y=cars_data1["Number of Forward Gears"],hue="Classification",data=cars_data1)
# plt.show()


#Plotting box whiskers plot and histogram in the same window

# f,(ax_box,ax_hist)=plt.subplots(2,gridspec_kw={"height_ratios":(.5,.5)})

# sns.set(style='darkgrid')
# sns.boxplot(cars_data1['Number of Forward Gears'],ax=ax_box)
# sns.histplot(cars_data1['Number of Forward Gears'],ax=ax_hist,kde=False,bins=5)
# plt.show()

#Pairwise plots

sns.set(style='darkgrid')
sns.pairplot(cars_data1,kind='scatter',hue='Number of Forward Gears')
plt.show()