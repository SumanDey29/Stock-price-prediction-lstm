import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import tensorflow as tf
from keras.models import load_model
import streamlit as st


def enforce_uppercase():
    user_input = st.session_state.stock_ticker.upper()
    return user_input
 

st.title('Stock Price Prediction')
st.session_state.stock_ticker = st.text_input('Enter Stock Ticker','AAPL') 
user_input = enforce_uppercase()
st.text("Current Stock Ticker: " + user_input)
ab = yf.Ticker(user_input)
df = ab.history(start = "2009-01-01", end = "2023-06-01", interval = "1d")


#describing data
st.subheader('Data of  '+user_input+ '  from 2009 - 2023')
st.write(df.describe())


st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 MovingAverage')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100,'r',label='MA100')
plt.plot(df.Close,'b')
plt.legend()
plt.show()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 MovingAverage & 200 MovingAverage')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100,'r',label='MA100')
plt.plot(ma200,'g',label='MA200')
plt.plot(df.Close,'b')
plt.legend()
plt.show()
st.pyplot(fig)





data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range = (0,1))
data_train_array = scaler.fit_transform(data_train)



model = load_model('keras_model.h5')


past_100_Days = data_train.tail(100)
final_df = past_100_Days.append(data_test , ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test, y_test=np.array(x_test),np.array(y_test)
y_predict = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predict = y_predict*scale_factor
y_test = y_test * scale_factor


st.subheader('Our Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predict,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)