# Importing the libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
    
#------------------------------------Data Preparation---------------------------------#

# Import Data Set
df1 = pd.read_csv('FRANCE.csv')

df1['Date']= df1["date"].astype(str) +"/"+ df1["month"].astype(str) + '/'+ df1["year"].astype(str)

#Taking care of missing values
cols = ['3']
df1[cols] = df1[cols].fillna(df1[cols].shift(1))
print (df1)
df1.isnull().sum()

#Sorting of data
df1 = df1.drop(['year', 'country', 'month', 'date' , 'cs'], axis=1)

# Put your inputs into a single list
input_cols = ['1','2','3','4','5','6','7','8','9','10','11','12',
              '13','14','15','16','17','18','19','20','21','22','23','24']

df1['single_input_vector'] = df1[input_cols].apply(tuple, axis=1).apply(list)

required_cols = ['Date','single_input_vector']
df2 = df1[required_cols]
 
lens = list(map(len, df2['single_input_vector'].values))
 

res = pd.DataFrame({'Date': np.repeat(
    df2['Date'], lens), 'single_input_vector': np.concatenate(df2['single_input_vector'].values)})
 
print(res)
#Adding Time dependencies for the data given 
Time1 = ['0:00:00','1:00:00','2:00:00','3:00:00','4:00:00','5:00:00','6:00:00','7:00:00','8:00:00','9:00:00','10:00:00',
       '11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00',
       '18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']

T = pd.DataFrame(Time1,columns=['Time'])
res = res.reset_index()
T = T.reset_index()
res = [res, T]
res_f = pd.concat(res, axis=1)

res_f = res_f.drop(['index','index'],axis = 1)

a = np.array([ ['0:00:00','1:00:00','2:00:00','3:00:00','4:00:00','5:00:00','6:00:00','7:00:00','8:00:00','9:00:00','10:00:00',
       '11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00',
       '18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']])
res_f['Time'] = np.resize(a, res_f.shape[0])

#Concatinating to create datetime format

res_f['DateTime']= res_f["Date"].astype(str) +" "+ res_f["Time"].astype(str) 

required_cols = ['DateTime','single_input_vector']
res_f = res_f[required_cols]
r_f = res_f.iloc[12518:,:]

#Importing the weather dataset
weather = pd.read_csv("weathernational_data.csv",infer_datetime_format=True, 
                     low_memory=False,parse_dates={'datetime':[0]},index_col=['datetime'])
weather = weather.iloc[:150259,:]
#Resampling weather in Hour on date time and preparing for concatenation 
weather = weather.resample('H').mean()

#Concatenating the weather and r_f dataset
r_f = r_f.reset_index()
weather = weather.reset_index()
r_f = [r_f, weather]
r_f = pd.concat(r_f, axis=1)

r_f = r_f.drop(['index','datetime'],axis = 1)

# Sorting data
data = [r_f["DateTime"], 
        r_f["single_input_vector"],
        r_f["outdoor_humidity"],
        r_f["outdoor_temperature"],
        r_f["wind_speed"],]
header= ['DateTime',
         'Global_active_power',
         'outdoor_humidity',
         "outdoor_temperature",
         "wind_speed"]
#converting strings into internal datetime 
r_f_sort = pd.concat(data, axis = 1, keys = header)


#convert these strings into internal datetimes
r_f_sort['DateTime'] = pd.to_datetime(r_f_sort['DateTime'])

#Break apart the date and get the year, month, week of year, day of month, hour
r_f_sort['Year']  = r_f_sort['DateTime'].dt.year
r_f_sort['Month'] = r_f_sort['DateTime'].dt.month
r_f_sort['Hour']  = r_f_sort['DateTime'].dt.hour
r_f_sort['DayofWeek'] = r_f_sort['DateTime'].dt.dayofweek
#Setting Indexing
r_f_sort = r_f_sort.set_index('DateTime')
#Splitting into required dataset

#Categorey Variable (year)  #Day(Date) Garbage -----Month Date Needs to be used as Categorical variable But its not f any signigicance---- 
from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
r_f_sort.iloc[:,4] = labelencoder.fit_transform(r_f_sort.iloc[:,4])
#---------------------------Preparing the data for RNN----------------------#

#Split to train and test
train_size = int(len(r_f_sort) * 0.99)
test_size = len(r_f_sort) - train_size
train, test = r_f_sort.iloc[0:train_size], r_f_sort.iloc[train_size:len(r_f)]
print(len(train), len(test))

#Feature Scaling of independent Variables
f_columns = ['outdoor_humidity', 'outdoor_temperature', 'wind_speed']
f_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:, f_columns] = f_transformer.transform(
  train[f_columns].to_numpy()
)
test.loc[:, f_columns] = f_transformer.transform(
  test[f_columns].to_numpy()
)


#Feature Scaling of dependent variables
cnt_transformer = RobustScaler()
cnt_transformer = cnt_transformer.fit(train[['Global_active_power']])
train['Global_active_power'] = cnt_transformer.transform(train[['Global_active_power']])
test['Global_active_power'] = cnt_transformer.transform(test[['Global_active_power']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 72
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train.Global_active_power, time_steps )
X_test, y_test = create_dataset(test, test.Global_active_power, time_steps)
print(X_train.shape, y_train.shape)

#Importing the keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()

#Adding the LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences= True , input_shape =(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))
#Adding the second layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))
#Adding a 3rd layer and some dropout regulariztion
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))
#Adding a 4th(last) layers and some dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer='adam', loss= 'mean_squared_error' )

#Fitting the RNN to the training set
regressor.fit(X_train , y_train ,batch_size = 72 ,  epochs = 30 )


#Prediction Plotting
y_pred = regressor.predict(X_test)

y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1,-1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1,-1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

#Plotting
plt.plot(y_test_inv.flatten(), label = 'true', marker = ".")
plt.plot(y_pred_inv.flatten(), label = 'predicted' , color = 'red', marker = ".")

plt.legend()
