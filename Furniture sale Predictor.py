from sklearn.feature_extraction.text import CountVectorizer # for feature exrtraction
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime 
from statsmodels.tsa.stattools import adfuller#Dicky fuller test
import numpy as np
import csv
import pandas as pd #for data processing and analysis
import matplotlib.pyplot as plt  #for data visualisation
from matplotlib.pylab import rcParams

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error   
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
dfb = pd.read_csv('Bookcases sorted .csv', index_col='Date',date_parser=dateparse)
#print(dfb.head())
"""
dfb['Date']=pd.to_datetime(dfb['Date'],infer_datetime_format=True) #to convert the date into standard form "yyyy-mm-dd"
dfbi=dfb.set_index('Date')# setting index  to divide the data "date " into regular time periods

"""
    #----------------------------------------   Checking Stationarity-----------------------------------------------------------------------------------------######################
tb=dfb['Sales']
	#---------------------------------------Plotting Time Series-------------------------------------------------------------------------------------------------------#####################
plt.title("Bookcases Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(tb)
plt.show()
    #---------------------------------------Function for testing stationarity---------------------------------------------------------
def test_stationarity(timeseries):
	print('Results of Dickey-Fuller Test:')
	dftest = adfuller(timeseries, autolag='AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)

test_stationarity(tb)

#
"""
Dicky Fuller test shows the stationarity of Bookcases.
#estimating trend
tb_log = np.log(tb)
plt.plot(tb_log)
plt.show(tb_log)
"""
from statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:    
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
--------------------------------# Chairs Details--------------------------------------------------------
dfc= pd.read_csv('chairs sorted.csv', index_col='Date',date_parser=dateparse)
#print(dfc.head())
"""
dfb['Date']=pd.to_datetime(dfb['Date'],infer_datetime_format=True) #to convert the date into standard form "yyyy-mm-dd"
dfbi=dfb.set_index('Date')# setting index  to divide the data "date " into regular time periods

"""
    #----------------------------------------   Checking Stationarity-----------------------------------------------------------------------------------------######################
tc=dfc['Sales']
	#---------------------------------------Plotting Time Series-------------------------------------------------------------------------------------------------------#####################
plt.title("Chairs Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(tc)
plt.show()

test_stationarity(tc)
#dicky fuller p value--o so stationarity occurs

rom statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:    
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')

#----------------------------------------------# Furnishings Details--------------------------------------------------------
dff= pd.read_csv('furnishings sorted.csv', index_col='Date',date_parser=dateparse)
tf=dff['Sales']
	#---------------------------------------Plotting Time Series-------------------------------------------------------------------------------------------------------#####################
plt.title("Furnishing Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(tf,'o')
plt.show()
test_stationarity(tf)

#dicky fuller p value--o so stationarity occurs
 rom statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:    
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
#----------------------------------------------# Tables Details--------------------------------------------------------
dft= pd.read_csv('Tables.csv', index_col='Date',date_parser=dateparse)
tt=dft['Sales']
	#---------------------------------------Plotting Time Series-------------------------------------------------------------------------------------------------------#####################
plt.title("Tables Sales plot")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.plot(tt,)
plt.show()
test_stationarity(tt)
#dicky fuller p value--o so stationarity occurs
rom statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:    
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')















"""stationarity of 4 sub categories founded."""