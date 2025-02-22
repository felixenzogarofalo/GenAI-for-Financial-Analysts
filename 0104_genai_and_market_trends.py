import backtrader as bt
import pandas as pd
import yfinance as yf
from transformers import pipeline

# Define the model name
model_name="distilbert-base-uncased-finetuned-sst-2-english"

# Fetch historical data from Alpha Vantage and filter by date
def fetch_data(symbol, start_date, end_date):
    # Download data using yfinance
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # If columns are a MultiIndex, flatten them:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Rename columns to lowercase so they match Backtrader's expectations
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    # Optionally drop the 'Adj Close' column if present
    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])
    
    # Ensure the DataFrame index is in datetime format and sorted in ascending order
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    
    return data

# Define a Backtrader strategy using the GenAI model
class GenAIStrategy(bt.Strategy):
    """
    A simple strategy that uses a free sentiment analysis model from Hugging Face.
    
    For each new bar, it:
      - Simulates a news headline based on the price movement.
      - Uses a Hugging Face sentiment analysis pipeline to determine sentiment.
      - Buys if the sentiment is positive (and not in a position) or sells if the sentiment is negative (and already in a position).
    """
    
    def __init__(self):
        self.order = None
        # Create the sentiment analysis pipeline.
        # Note: The first call may download the model, which could take a while.
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
        )

    def next(self):
        # Get sentiment: 1 for positive, -1 for negative, 0 for neutral.
        sentiment = self.get_sentiment()
        self.log(f"Sentiment: {sentiment}")

        # Trading logic: Buy on positive sentiment if not already in the market; sell on negative sentiment if in the market.
        if sentiment > 0 and not self.position:
            self.order = self.buy()
            self.log(f"BUY ORDER CREATED at {self.data.close[0]:.2f}")
        elif sentiment < 0 and self.position:
            self.order = self.sell()
            self.log(f"SELL ORDER CREATED at {self.data.close[0]:.2f}")

    def get_sentiment(self):
        """
        Simulate a news headline based on price movement and use a Hugging Face sentiment analysis model.
        Returns:
            int: 1 for positive sentiment, -1 for negative sentiment, 0 for neutral or error.
        """
        # Ensure there are at least two bars to compare.
        if len(self.data) < 2:
            return 0

        # Create a simulated headline based on today's close relative to yesterday's.
        if self.data.close[0] > self.data.close[-1]:
            headline = "Stock surges as investors remain optimistic"
        else:
            headline = "Stock declines amid investor concerns"

        try:
            # Analyze the headline sentiment.
            result = self.sentiment_analyzer(headline)[0]
            label = result.get('label', '').upper()
            self.log(f"Headline: '{headline}' -> {label}")
            
            if label == "POSITIVE":
                return 1
            elif label == "NEGATIVE":
                return -1
            else:
                return 0
        except Exception as e:
            self.log("Sentiment analysis error: " + str(e))
            return 0

    def log(self, txt, dt=None):
        """Logging function for this strategy."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

# Set backtest date range
start_date = "2023-01-01"
end_date = "2024-01-01"

# Fetch Alpha Vantage data
symbol = "SPY"
data = fetch_data(symbol, start_date, end_date)

# Convert to Backtrader format
data_feed = bt.feeds.PandasData(dataname=data)

# Initialize Backtrader
cerebro = bt.Cerebro()

# Add our ChatGPT-based strategy
cerebro.addstrategy(GenAIStrategy)

# Set our starting cash
cerebro.broker.setcash(10000.0)

# Load historical data using Yahoo Finance
cerebro.adddata(data_feed)

# Print the starting portfolio value
print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

# Run the backtest
cerebro.run()

# Print the final portfolio value
print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

# Optionally, plot the results
cerebro.plot()
