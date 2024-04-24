from flask import Flask, render_template, request, flash, redirect, url_for
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import re
import nltk
import statsmodels.api as sm
import yfinance as yf
import preprocessor as p
import constants as ct
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math, random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
nltk.download('punkt')
import warnings
import webbrowser
from utils import recommending, LR, ARIMA, LSTM, retrieving_news_polarity, clean_text

app = Flask(__name__)



@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
   return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    print("caaled")
    input1=request.form.get("companysymbol")
    print(input1)
    end = datetime.now()
    start = datetime(end.year-2,end.month,end.day)
    data = yf.download(input1, start=start, end=end)
    if(data.empty):
        return render_template("Error.html")
    data_frame = pd.DataFrame(data=data)
    data_frame.to_csv(''+input1+'.csv')
    data_frame=pd.read_csv(''+input1+'.csv')
    today_stock=data_frame.iloc[-1:]
    data_frame = data_frame.dropna()
    code_list=[]
    for i in range(0,len(data_frame)):
        code_list.append(input1)
    data_frame2=pd.DataFrame(code_list,columns=['Code'])
    data_frame2=pd.concat([data_frame2, data_frame], axis=1)
    data_frame=data_frame2
    print(data_frame)
    data_frame, lr_prediction,lr_error=LR(data_frame,input1)
    arima_prediction, arima_error=ARIMA(data_frame,input1)
    lstm_prediction, lstm_error=LSTM(data_frame,input1)
    polarity,tweet_list,tweet_polarity,pos,neg,neutral = retrieving_news_polarity("888c604dc40544ff992fd30a60cf6998",input1)
    mean=0.25*lr_prediction+0.25*lstm_prediction+0.5*arima_prediction
    idea, decision=recommending(data_frame, polarity,today_stock,mean,input1)
    return render_template('results.html',input1=input1,arima_prediction=round(arima_prediction,2),lstm_prediction=round(lstm_prediction,2),
                               lr_prediction=round(lr_prediction,2),open_stock=today_stock['Open'].to_string(index=False),
                               close_stock=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                               tweet_list=tweet_list,tweet_polarity=tweet_polarity,idea=idea,decision=decision,high_stock=today_stock['High'].to_string(index=False),
                               low_stock=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False),
                               lr_error=round(lr_error,2),lstm_error=round(lstm_error,2),arima_error=round(arima_error,2))






if __name__ == '__main__':
   webbrowser.open('http://127.0.0.1:5000/')
   app.run(port='5000')