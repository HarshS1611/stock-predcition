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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") 
import math, random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
nltk.download('punkt')
import warnings

def LR(data_frame, input1):

    forecast_out = int(7)
    data_frame['Close after n days'] = data_frame['Close'].shift(-forecast_out)
    data_frame_new = data_frame[['Close', 'Close after n days']]

    y = np.array(data_frame_new.iloc[:-forecast_out, -1])
    y = np.reshape(y, (-1, 1))

    X = np.array(data_frame_new.iloc[:-forecast_out, 0:-1])

    X_to_be_forecasted = np.array(data_frame_new.iloc[-forecast_out:, 0:-1])

    X_train = X[0:int(0.8 * len(data_frame)), :]
    X_test = X[int(0.8 * len(data_frame)):, :]
    y_train = y[0:int(0.8 * len(data_frame)), :]
    y_test = y[int(0.8 * len(data_frame)):, :]

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_to_be_forecasted = sc.transform(X_to_be_forecasted)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    y_test_prediction = clf.predict(X_test)
    y_test_prediction = y_test_prediction * (1.04)
    import matplotlib.pyplot as plot
    fig = plot.figure(figsize=(7.2,4.8),dpi=65)
    plot.plot(y_test,label='Actual Price' )
    plot.plot(y_test_prediction,label='Predicted Price')
        
    plot.legend(loc=4)
    plot.savefig('static/LR.png')
    plot.close(fig)

    lr_error = math.sqrt(mean_squared_error(y_test, y_test_prediction))

    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set * (1.04)
    lr_prediction = forecast_set[0, 0]

    print()
    print("Tomorrow's ", input1, " Closing Price Prediction by Linear Regression: ", lr_prediction)
    print("Linear Regression RMSE:", lr_error)
    return data_frame, lr_prediction, lr_error


def ARIMA(data_frame, input1):
    uniqueVals = data_frame["Code"].unique()
    len(uniqueVals)
    data_frame = data_frame.set_index("Code")

    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d')

    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = sm.tsa.arima.ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions

    for company in uniqueVals[:10]:
        data = (data_frame.loc[company, :]).reset_index()
        data['Price'] = data['Close']
        Quantity_date = data[['Price', 'Date']]
        Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
        Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['Date'], axis=1)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(Quantity_date)
        plt.savefig('static/Trends.png')
        plt.close(fig)
        plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(data_frame['Date'], data_frame['Close'], label='Closing Price')
        plt.title(f'{input1} Stock Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.savefig('static/StockPriceTrend.png')

        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]

        predictions = arima_model(train, test)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(test,label='Actual Price')
        plt.plot(predictions,label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/ARIMA.png')
        plt.close(fig)
        print()

        arima_prediction = predictions[-2]
        print("Tomorrow's", input1, " Closing Price Prediction by ARIMA:", arima_prediction)
        arima_error = math.sqrt(mean_squared_error(test, predictions))
        print("ARIMA RMSE:", arima_error)
        return arima_prediction, arima_error


def LSTM(data_frame, input1):
    dataset_train = data_frame.iloc[0:int(0.8 * len(data_frame)), :]
    dataset_test = data_frame.iloc[int(0.8 * len(data_frame)):, :]

    training_set = data_frame.iloc[:, 4:5].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 7:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_forecast = np.array(X_train[-1, 1:])
    X_forecast = np.append(X_forecast, y_train[-1])

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LSTM

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=25, batch_size=32)

    real_stock_price = dataset_test.iloc[:, 4:5].values

    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_set = testing_set.reshape(-1, 1)

    testing_set = sc.transform(testing_set)

    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i - 7:i, 0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = regressor.predict(X_test)

    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    fig = plt.figure(figsize=(7.2,4.8),dpi=65)
    plt.plot(real_stock_price,label='Actual Price')  
    plt.plot(predicted_stock_price,label='Predicted Price')
          
    plt.legend(loc=4)
    plt.savefig('static/LSTM.png')
    plt.close(fig)

    lstm_error = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

    forecasted_stock_price = regressor.predict(X_forecast)

    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

    lstm_prediction = forecasted_stock_price[0, 0]
    print("Tomorrow's ", input1, " Closing Price Prediction by LSTM: ", lstm_prediction)
    print("LSTM RMSE:", lstm_error)
    return lstm_prediction, lstm_error

import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Remove mentions (@)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # Remove hashtags (#)
    text = re.sub(r'#', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

def retrieving_news_polarity(api_key, query, num_articles=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": num_articles
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Error fetching news articles:", response.json())
        return None

    articles = response.json().get("articles", [])
    if not articles:
        print("No articles found for the query:", query)
        return None

    news_list = []
    global_polarity = 0
    twe_list=[]
    pos=0
    neg=0
    neutral=0

    for article in articles:
        title = article.get("title", "")
        news_list.append(title)

        # Preprocess the article title
        tw = clean_text(title)

        # Calculate polarity of the article title
        blob = TextBlob(tw)
        polarity = 0
        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity
            if polarity > 0:
                pos += 1
            elif polarity < 0:
                neg += 1

            global_polarity += sentence.sentiment.polarity

        twe_list.append(title)

    # Calculate overall polarity
    if news_list:
        global_polarity /= len(news_list)

    # Calculate neutral count
    neutral = num_articles - pos - neg
    if neutral < 0:
        neg += neutral
        neutral = 0

    # Plot pie chart
    labels=['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    explode = (0, 0, 0)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.tight_layout()
    plt.savefig('static/SA.png')
    plt.close(fig)

    # Determine overall polarity label
    news_polarity = "Overall Positive" if global_polarity > 0 else "Overall Negative"

    return global_polarity, news_list[:11], news_polarity, pos, neg, neutral



def recommending(data_frame, global_polarity, today_stock, mean, input1):
    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity > 0:
            idea = "RISE"
            decision = "BUY"
            print()
            print("According to the LR, ARIMA, LSTM Predictions and Sentiment Analysis of newss, a", idea,
                  "in", input1,
                  "stock is expected, so you should buy", decision, "it")
        elif global_polarity <= 0:
            idea = "FALL"
            decision = "SELL"
            print()
            print("According to the LR, ARIMA, LSTM Predictions and Sentiment Analysis of newss, a", idea,
                  "in", input1,
                  "stock is expected, so you should buy", decision, "it")
    else:
        idea = "FALL"
        decision = "SELL"
        print()
        print("According to the LR, ARIMA, LSTM Predictions and Sentiment Analysis of newss, a", idea,
              "in", input1,
              "stock is expected, so you should buy", decision, "it")
    return idea, decision