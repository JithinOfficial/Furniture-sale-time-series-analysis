from sklearn.feature_extraction.text import CountVectorizer # for feature exrtraction
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime 
from statsmodels.tsa.stattools import adfuller#Dicky fuller test
import numpy as np
import csv
import pandas as pd #for data processing and analysis
import matplotlib.pyplot as plt  #for data visualisation
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6


"""

#------------------------------------------------------------ ### Data checking--------------------------------------------------------------------------------------------------------------------------------------###########
df=pd.read_csv('oridata.csv') #(from sample data required data is extracted('Order Date', 'State', 'Category', 'Sub-Category', 'Product Name','Sales', 'Quantity') and its encoding also changed)
#print(df.head())
#print(df.columns)
#print(df.describe())   
#print(df.index) 
#----------------------------------- data cleaning ('Order Date', 'State', 'Category', 'Sub-Category', 'Product Name','Sales', 'Quantity')-- ---------------------------------------------------------------------###########
i did data cleaning in excel its little bit easy than python as we can reduce type conversion errors
#df=df[df.Category=="Furniture"]  #3from the given data Furniture Category only selected as our aim is to analyse Furniture sales only
"""









#----------------------------------------------# Bookcases Details--------------------------------------------------------
dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y')
dfc = pd.read_csv('Bookcases sorted .csv', index_col='Date',date_parser=dateparse)
#df=df[df.Category=="Furniture"]  #3from the given data Furniture Category only selected as our aim is to analyse Furniture sales only



#----------------------------------to find sub categories count---------------------------------------------------------------------------------------------------------------------------------------------------############


 

dfc['Date']=pd.to_datetime(dfc['Date'],infer_datetime_format=True) #to convert the date into standard form "yyyy-mm-dd"
dfci=dfc.set_index('Date')# setting index  to divide the data "date " into regular time periods


plt.title("Table Sales plot")
plt.title("Chair Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(dfci.index,dfci['Sales'],'o')
#plt.show()
#__________________________________to find the  sales is stationary or not______________________________________#



#### Determining rolling statistics 
romc=dfci.rolling(window=12).mean()
rosc=dfci.rolling(window=12).std()
#print(romc,rosc)

# plotting rolling statistics
origc=plt.plot(dfci.index,dfci['Sales'],'o',color="blue",label="original")
meanc=plt.plot(romc,'o',color='red',label="Rolling Mean")
stdc=plt.plot(rosc,'o',color='black',label="Rolling std")
plt.legend(loc='best')
plt.title("Rolling mean &Standard Deviation of Chairs sales")
#plt.show()
"""
#_____________________________________ANother method for stationary check_________________________________#############
"""

"""
#--------------------------------------------------------------------------------------------------------Trend Estimation------------------------------------------------------------------------------------------------------------################


# trend estimation ###############







 
#---------------------------------------------- # Tables Detaiils----------------------------------------------------------
dft=df[df.Sub=="Tables"] 

dft=pd.concat([dft['Date'],dft['Sales']],axis=1,sort=False)


print(dft.tail())
dft['Date']=pd.to_datetime(dft['Date'],infer_datetime_format=True) #to convert the date into standard form "yyyy-mm-dd"
dfti=dft.set_index(['Date'])# setting index  to divide the data "date " into regular time periods
plt.title("Table Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(dfti.index,dfti['Sales'],'o')
plt.show()

#### Determining rolling statistics 
romt=dfti.rolling(window=12).mean()
rost=dfti.rolling(window=12).std()
#print(romt,rost)

# plotting rolling statistics
origt=plt.plot(dfti.index,dfti['Sales'],'o',color="blue",label="original")
meant=plt.plot(romt,'o',color='red',label="Rolling Mean")
stdt=plt.plot(rost,'o',color='black',label="Rolling std")
plt.legend(loc='best')
plt.title("Rolling mean &Standard Deviation of Tables sales")
plt.show()




#----------------------------------------------# Bookcases Details--------------------------------------------------------
dfb=df[df.Sub=="Bookcases"]
dfb=pd.concat([dfb['Date'],dfb['Sales']],axis=1,sort=False)

dfb['Date']=pd.to_datetime(dfb['Date'],infer_datetime_format=True) #to convert the date into standard form "yyyy-mm-dd"
dfbi=dfb.set_index(['Date'])# setting index  to divide the data "date " into regular time periods
plt.title("Bookcases Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(dfbi.index,dfbi['Sales'],'o')
plt.show()


#### Determining rolling statistics 
romb=dfbi.rolling(window=12).mean()
rosb=dfbi.rolling(window=12).std()
#print(romb,rosb)

# plotting rolling statistics
origb=plt.plot(dfbi.index,dfbi['Sales'],'o',color="blue",label="original")
meanb=plt.plot(romb,'o',color='red',label="Rolling Mean")
stdb=plt.plot(rosb,'o',color='black',label="Rolling std")
plt.legend(loc='best')
plt.title("Rolling mean &Standard Deviation of Bookcases sales")
plt.show()




#---------------------------------------------- # Furnishings Details----------------------------------------------------
dff=df[df.Sub=="Furnishings"]
dff=pd.concat([dff['Date'],dff['Sales']],axis=1,sort=False)

dff['Date']=pd.to_datetime(dff['Date'],infer_datetime_format=True) #to convert the date into standard form "yyyy-mm-dd"
dffi=dff.set_index(['Date'])# setting index  to divide the data "date " into regular time periods
plt.title("Furnishings Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(dffi.index,dffi['Sales'],'o')
plt.show()


#### Determining rolling statistics 
romf=dffi.rolling(window=12).mean()
rosf=dffi.rolling(window=12).std()
#print(romf,rosf)

# plotting rolling statistics
origf=plt.plot(dffi.index,dffi['Sales'],'o',color="blue",label="original")
meanf=plt.plot(romf,'o',color='red',label="Rolling Mean")
stdf=plt.plot(rosf,'o',color='black',label="Rolling std")
plt.legend(loc='best')
plt.title("Rolling mean &Standard Deviation of Furnishings sales")
plt.show()
"""

