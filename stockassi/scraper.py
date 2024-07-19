import yfinance as yf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from models import db, Stock
from datetime import datetime
import pandas as pd
import time

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    
    # Fetch current price
    price = stock.history(period='1d')['Close'].iloc[-1]
    
    # Fetch stock info for market cap, 52-week high, and 52-week low
    info = stock.info
    market_cap = info.get('marketCap', 'N/A')
    week_high = info.get('fiftyTwoWeekHigh', 'N/A')
    week_low = info.get('fiftyTwoWeekLow', 'N/A')

    # Fetch historical data for the last 5 days
    hist = stock.history(period='5d')
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(hist)
    
    # Save data to database
    stock_db = Stock.query.filter_by(ticker=ticker).first()
    if stock_db:
        stock_db.price = price
        stock_db.market_cap = market_cap
        stock_db.week_high = week_high
        stock_db.week_low = week_low
        stock_db.history = hist.to_csv()
        stock_db.last_updated = datetime.utcnow()
    else:
        stock_db = Stock(ticker=ticker, price=price, market_cap=market_cap, week_high=week_high, week_low=week_low, history=hist.to_csv())
        db.session.add(stock_db)
    
    db.session.commit()
    
    return price, market_cap, week_high, week_low, indicators, hist

def calculate_technical_indicators(hist):
    indicators = {}

    # Calculate SMA
    indicators['SMA'] = hist['Close'].rolling(window=20).mean().iloc[-1]

    # Calculate EMA
    indicators['EMA'] = hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]

    # Calculate RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs)).iloc[-1]

    # Calculate MACD
    macd = hist['Close'].ewm(span=12, adjust=False).mean() - hist['Close'].ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    indicators['MACD'] = macd.iloc[-1]
    indicators['MACD Signal'] = signal.iloc[-1]

    return indicators

def fetch_stock_news(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        time.sleep(5)
        
        news_elements = driver.find_elements(By.CSS_SELECTOR, 'div.Cf')
        news_content = "\n\n".join([elem.text for elem in news_elements if elem.text])
        
        return news_content if news_content else "No news available"
    except Exception as e:
        print(f"Error: {e}")
        return "Failed to fetch news"
    finally:
        driver.quit()
