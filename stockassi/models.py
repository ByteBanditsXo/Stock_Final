import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set the predefined URL
PREDEFINED_URL = 'https://finance.yahoo.com/'

def fetch_stock_data(ticker):
    url = PREDEFINED_URL + ticker
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    
    time.sleep(5)
    
    stock_data = extract_stock_data(driver)
    driver.quit()
    
    return stock_data

def extract_stock_data(driver):
    stock_data = {}
    try:
        stock_data['price'] = float(driver.find_element(By.CSS_SELECTOR, 'fin-streamer[data-field="regularMarketPrice"]').text.replace(',', '').replace('$', ''))
        historical_prices_elements = driver.find_elements(By.CSS_SELECTOR, 'fin-streamer[data-field="regularMarketPreviousClose"]')  # Adjust selector as needed
        prices = [float(price.text.replace(',', '').replace('$', '')) for price in historical_prices_elements]
        stock_data['prices'] = prices
        stock_data['shares_outstanding'] = float(driver.find_element(By.CSS_SELECTOR, 'td[data-test="SHARES_OUTSTANDING-value"]').text.replace(',', ''))  # Adjust selector as needed
    except Exception as e:
        print(f"An error occurred while extracting stock data: {e}")
    return stock_data

def calculate_sma(prices, window):
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(prices, weights, mode='full')[:len(prices)]
    a[:window] = a[window]
    return a[-1]

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1.+rs)

    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi[-1]

def calculate_macd(prices, n_fast=12, n_slow=26):
    ema_fast = calculate_ema(prices, n_fast)
    ema_slow = calculate_ema(prices, n_slow)
    return ema_fast - ema_slow

def calculate_market_cap(price, shares_outstanding):
    return price * shares_outstanding

def get_financial_data(ticker):
    return yf.Ticker(ticker).history(period="1y")

def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean().iloc[-1]

def calculate_ema(data, window=20):
    return data['Close'].ewm(span=window, adjust=False).mean().iloc[-1]

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period-1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period-1, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    return rsi

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

def calculate_market_cap(ticker):
    stock = yf.Ticker(ticker)
    market_price = stock.history(period='1d')['Close'].iloc[-1]
    shares_outstanding = stock.info['sharesOutstanding']
    market_cap = market_price * shares_outstanding
    return market_cap

def predict_stock_price(data, days_ahead):
    data.reset_index(inplace=True, drop=False)
    data['Days'] = data.index
    X = np.array(data['Days']).reshape(-1, 1)
    y = np.array(data['Close'])
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.array([len(data) + i for i in range(days_ahead)]).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions.tolist()

def plot_predictions(ticker, days_ahead):
    data = get_financial_data(ticker)
    predictions = predict_stock_price(data, days_ahead)

    # Print predicted prices
    predicted_prices = []
    for i, price in enumerate(predictions, 1):
        predicted_prices.append(f"Day {i}: ${price:.2f}")

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Historical Prices')
    future_dates = pd.date_range(start=data.index[-1], periods=days_ahead+1, closed='right')
    plt.plot(future_dates, predictions, label='Predicted Prices', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plot_filename = f'static/{ticker}_prediction.png'
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename, predicted_prices