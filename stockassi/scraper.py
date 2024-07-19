import yfinance as yf
import pandas as pd
from playwright.sync_api import sync_playwright

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    info = stock.info

    hist['SMA'] = hist['Close'].rolling(window=20).mean()
    hist['EMA'] = hist['Close'].ewm(span=20, adjust=False).mean()
    hist['RSI'] = compute_rsi(hist['Close'])
    hist['MACD'], hist['MACD_signal'] = compute_macd(hist['Close'])

    return {
        'current_price': info.get('currentPrice', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
        'sma': hist['SMA'].iloc[-1],
        'ema': hist['EMA'].iloc[-1],
        'rsi': hist['RSI'].iloc[-1],
        'macd': hist['MACD'].iloc[-1],
        'macd_signal': hist['MACD_signal'].iloc[-1]
    }

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

def fetch_stock_news(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_selector('div.Cf')
        content = page.content()
        browser.close()
        return content
    


from playwright.sync_api import sync_playwright

def fetch_stock_news(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        
        # Wait for the entire page to load
        page.wait_for_load_state('networkidle')  # Wait until network is idle
        
        try:
            # Wait for a general element or specific section
            page.wait_for_selector('section', timeout=60000)  # Increase timeout if needed
            print("Selector found")
        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path='screenshot.png')  # Take a screenshot for debugging
            browser.close()
            return "Failed to fetch news"

        content = page.content()
        browser.close()
        return content
    
    # scraper.py

from playwright.sync_api import sync_playwright

def fetch_stock_news(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Increase timeout to 60 seconds
            page.goto(url, timeout=60000)  
            page.wait_for_load_state('networkidle', timeout=60000)  
            print("Page loaded successfully")
            
            # Increase timeout for waiting for selector
            page.wait_for_selector('section', timeout=60000)  
            print("Selector found")
        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path='screenshot.png')  # Take a screenshot if it fails
            browser.close()
            return "Failed to fetch news"

        content = page.content()
        browser.close()
        return content
# Add other functions or code here if needed