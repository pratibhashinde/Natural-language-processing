import pandas_datareader as pdr

#loading data
#data is taken from ttingo website.login to tiingo, go to api and then copy api key from authentication section.
# apple company stock data.
df=pdr.get_data_tiingo('AAPL',api_key='f697c129af9fdfc834a05a340f794f1acafe15be')
df.to_csv('data.csv')

import pandas as pd
df=pd.read_csv('data.csv')  #stock prices as input

import matplotlib.pyplot as plt
plt.plot(df['close'])

#minmaxscaler---as LSTM is sensitive to values
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df1=df['close']
obj=MinMaxScaler(feature_range=(0,1))
df1=obj.fit_transform(np.array(df1).reshape(-1,1))

#train-test data 65-35
train_size=int(len(df1)*.65)
test_size=len(df1)-train_size

train=df1[0:train_size,0]
test=df1[train_size:len(df1),0]

#create dataset from stock price values by taking a timestep
#if we consider a timestep of 3, means value at time =4 is obtained by considering time steps 1,2,3.Similarly goes on

def create_data(dataset,timestep):
    x,y=[],[]
    for i in range(len(dataset)-timestep):
        a=dataset[i:(i+timestep)]
        x.append(a)
        y.append(dataset[i+timestep])
    return np.array(x),np.array(y)

timestep=100
x_train,y_train=create_data(train,timestep)
x_test,y_test=create_data(test,timestep)

#reshape data into 3D which is needed for LSTM
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#create stacked LSTM model-each layer has 50 lstm cells/units
from keras.layers import Dense,LSTM
from keras.models import Sequential
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(timestep,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)

#prediction and evaluation
y_pred=model.predict(x_test)

#rescaling prediction values
y_pred1=obj.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error
import math
error=math.sqrt(mean_squared_error(y_pred1,y_test))

#plot the result
y_pred_train=model.predict(x_train)
x=[i for i in range(0,len(train)+len(test))]
x1=[i for i in range(timestep,len(train))]
x2=[i for i in range(len(train)+timestep,len(train)+len(test))]

plt.plot(x1,obj.inverse_transform(y_pred_train),c='k')
plt.plot(x2,y_pred1,c='r')
plt.plot(df['close'],c='b')

#predict stock price for next x number of days
x_input=test[len(y_test):].reshape(1,-1)

input_list=list(x_input)
input_list1=input_list[0].tolist()
days=30
output_pred=[]
for i in range(0,days):
    ar1=np.array(input_list1[i:])
    #print(ar1[0:3])
    in1=ar1.reshape(1,timestep,1)
    y_pred=model.predict(in1)
    output_pred.append((list(y_pred))[0].tolist())
    input_list1.append((list(y_pred))[0][0].tolist())

#plot complete stock price

y1=[i for i in range(len(train)+len(test)-timestep,len(train)+len(test))]
y2=[i for i in range(len(train)+len(test),len(train)+len(test)+days)]

plt.plot(y1,obj.inverse_transform(np.array(input_list1[0:timestep]).reshape(-1,1)),c='b')
plt.plot(y2,obj.inverse_transform(output_pred),c='r')

#another result plot
df_new=df1.tolist()
df_new.extend(output_pred)
plt.plot(df_new[1200:])