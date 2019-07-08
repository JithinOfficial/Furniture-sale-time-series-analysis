from sklearn.feature_extraction.text import CountVectorizer # for feature exrtraction
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime 
from statsmodels.tsa.stattools import adfuller#Dicky fuller test
import numpy as np
import pandas as pd #for data processing and analysis
import matplotlib.pyplot as plt  #for data visualisation
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6

dfc=pd.read_csv('oridata.csv') #(from sample data required data is extracted('Order Date', 'State', 'Category', 'Sub-Category', 'Product Name','Sales', 'Quantity') and its encoding also changed)


#print(df.columns)
#------------------------------------------------------------ ### Data checking#########--------------------------------------------------------------------------------------------------------------------------###########
#print(df.head())
#print(df.describe())   
#print(df.columns)
#print(df.index)
#----------------------------------- data cleaning ('Order Date', 'State', 'Category', 'Sub-Category', 'Product Name','Sales', 'Quantity')-- ---------------------------------------------------------------------###########

dfc=dfc[dfc.Category=="Furniture"] 

dfc=pd.concat([dfc['Date'],dfc['Sales']],axis=1,sort=False)
#print(dfc)

dfc['Date']=pd.to_datetime(dfc['Date'],infer_datetime_format=True) #to convert the date into standard form "yyyy-mm-dd"
dfci=dfc.set_index(['Date'])# setting index  to divide the data "date " into regular time periods
print(dfci)

plt.title("Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(dfci.index,dfci['Sales'],'o')
#plt.show()

#### Determining rolling statistics 
romc=dfci.rolling(window=12).mean()
rosc=dfci.rolling(window=12).std()
#print(romc,rosc)

# plotting rolling statistics
origc=plt.plot(dfci.index,dfci['Sales'],'o',color="blue",label="original")
meanc=plt.plot(romc,'o',color='red',label="Rolling Mean")
stdc=plt.plot(rosc,'o',color='black',label="Rolling std")
plt.legend(loc='best')
plt.title("Rolling mean &Standard Deviation")
#plt.show()


#  Dicky Fuller Test
print("Results of Dicky Fuller Test:")



adfuller(timeseries, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

c=comb(dfci['Sales'])



dfcpc=pd.Series(c[0:4],index=['Test Statistic:','p-value:','#Lags Used','No. Of Observations Used:'])
 



for key,value in dftc[4].items():
	dfopc['Critical Value(%s)'%key]=value
print(dfopc)