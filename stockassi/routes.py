from flask import Blueprint, request, render_template
from models import db
from scraper import fetch_stock_data, fetch_stock_news

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main_routes.route('/stock', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return render_template('index.html', error='Ticker symbol is required')

    try:
        # Fetch stock data
        price, market_cap, week_high, week_low, indicators, hist = fetch_stock_data(ticker)
        news_content = fetch_stock_news(ticker)

        # Format historical data
        history_html = hist.to_html(classes='data')

        return render_template('index.html', 
                               ticker=ticker, 
                               price=price, 
                               market_cap=market_cap, 
                               week_high=week_high, 
                               week_low=week_low, 
                               indicators=indicators, 
                               history=history_html, 
                               news=news_content)
    except Exception as e:
        return render_template('index.html', error=str(e))

def configure_routes(app):
    app.register_blueprint(main_routes)

