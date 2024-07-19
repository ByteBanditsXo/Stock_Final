from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize the SQLAlchemy instance
db = SQLAlchemy()

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False, unique=True)
    price = db.Column(db.Float, nullable=True)
    market_cap = db.Column(db.BigInteger, nullable=True)
    week_high = db.Column(db.Float, nullable=True)
    week_low = db.Column(db.Float, nullable=True)
    history = db.Column(db.Text, nullable=True)  # Store historical data in CSV format
    news_content = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<Stock {self.ticker}>"
    
    def to_dict(self):
        return {
            'ticker': self.ticker,
            'price': self.price,
            'market_cap': self.market_cap,
            'week_high': self.week_high,
            'week_low': self.week_low,
            'history': self.history,
            'news_content': self.news_content,
            'last_updated': self.last_updated.strftime('%Y-%m-%d %H:%M:%S')
        }
