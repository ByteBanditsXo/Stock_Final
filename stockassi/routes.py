from flask import Blueprint, request, jsonify, render_template_string
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

main_routes = Blueprint('main_routes', __name__)

# Fetch financial data for a given ticker
def get_financial_data(ticker):
    return yf.Ticker(ticker).history(period="1y")

# Calculate Simple Moving Average (SMA)
def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean().iloc[-1]

# Calculate Exponential Moving Average (EMA)
def calculate_ema(data, window=20):
    return data['Close'].ewm(span=window, adjust=False).mean().iloc[-1]

# Calculate Relative Strength Index (RSI)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period-1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period-1, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    return rsi

# Calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

# Calculate Market Capitalization
def calculate_market_cap(ticker):
    stock = yf.Ticker(ticker)
    market_price = stock.history(period='1d')['Close'].iloc[-1]
    shares_outstanding = stock.info['sharesOutstanding']
    market_cap = market_price * shares_outstanding
    return market_cap

# Predict stock price
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

# Plot predicted stock prices
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

@ main_routes.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Market Assistant</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container">
                <h1 class="mt-4">Stock Market Assistant</h1>
                <div class="form-group">
                    <label for="ticker">Enter stock ticker:</label>
                    <input type="text" id="ticker" class="form-control" placeholder="e.g., AAPL, MSFT, TSLA">
                </div>
                <div class="form-group">
                    <label for="days_ahead">Days ahead to predict:</label>
                    <input type="number" id="days_ahead" class="form-control" value="30">
                </div>
                <button id="predictButton" class="btn btn-primary">Predict</button>
                <h2 class="mt-4">Prediction:</h2>
                <div id="response" class="mt-3"></div>
            </div>

            <script>
                document.getElementById('predictButton').addEventListener('click', async () => {
                    const ticker = document.getElementById('ticker').value;
                    const days_ahead = document.getElementById('days_ahead').value;
                    const responseDiv = document.getElementById('response');
                    responseDiv.innerHTML = '';

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ticker, days_ahead })
                    });

                    const result = await response.json();

                    if (result.error) {
                        responseDiv.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                    } else {
                        responseDiv.innerHTML = `
                            <h3>${result.message}</h3>
                            <ul>
                                ${result.predicted_prices.map(price => `<li>${price}</li>`).join('')}
                            </ul>
                            <img src="${result.plot_url}" class="img-fluid">
                            <h4>Additional Data</h4>
                            <ul>
                                <li>SMA: ${result.sma}</li>
                                <li>EMA: ${result.ema}</li>
                                <li>RSI: ${result.rsi}</li>
                                <li>Market Cap: ${result.market_cap}</li>
                                <li>MACD: ${result.macd}</li>
                                <li>MACD Signal: ${result.macd_signal}</li>
                            </ul>`;
                    }
                });
            </script>
        </body>
        </html>
    ''')

@main_routes.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker').upper()
    days_ahead = int(data.get('days_ahead'))

    try:
        data = get_financial_data(ticker)
        sma = calculate_sma(data)
        ema = calculate_ema(data)
        rsi = calculate_rsi(data)
        market_cap = calculate_market_cap(ticker)
        macd, macd_signal = calculate_macd(data)
        plot_url, predicted_prices = plot_predictions(ticker, days_ahead)
        
        return jsonify({
            'message': f'Predicted prices for {ticker} for the next {days_ahead} days:',
            'plot_url': plot_url,
            'predicted_prices': predicted_prices,
            'sma': f"${sma:.2f}",
            'ema': f"${ema:.2f}",
            'rsi': f"{rsi:.2f}",
            'market_cap': f"${market_cap:.2f}",
            'macd': f"${macd:.2f}",
            'macd_signal': f"${macd_signal:.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)})