# test_scraper.py

from scraper import fetch_stock_news

if __name__ == "__main__":
    ticker = 'AAPL'
    news = fetch_stock_news(ticker)
    print(news)