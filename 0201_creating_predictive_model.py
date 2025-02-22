import yfinance as yf
import backtrader as bt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------------
# 1. Data Fetching Function
# ----------------------------------------------------------------------------
def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical data using yfinance.
    Rename columns to match Backtrader's expectations.
    Return a pandas DataFrame sorted by date.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # If columns are a MultiIndex, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Rename columns
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    # Drop 'Adj Close' if present
    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])
    
    # Ensure the index is datetime and sorted
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    
    return data

# ----------------------------------------------------------------------------
# 2. Get Two Years of Data (2023 for training, 2024 for testing)
# ----------------------------------------------------------------------------
df = fetch_data('AAPL', '2023-01-01', '2024-01-01')

# We will train on 2023 and test (live) on 2024 in Backtrader
train_df = df.loc[:'2023-12-31'].copy()
test_df = df.loc['2024-01-01':'2025-01-01'].copy()

# ----------------------------------------------------------------------------
# 3. Prepare Training Data
#    We do feature engineering on the training set to train our model.
# ----------------------------------------------------------------------------
train_df['return'] = train_df['close'].pct_change()
train_df['ma10'] = train_df['close'].rolling(window=10).mean()
train_df.dropna(inplace=True)

# Define features (X) and label (y)
X_train = train_df[['open', 'high', 'low', 'close', 'volume', 'ma10']]
y_train = (train_df['return'] > 0).astype(int)

# ----------------------------------------------------------------------------
# 4. Train the Random Forest Model
# ----------------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 5. Backtrader Strategy: Online Predictions
#    - For each new bar, gather the last 20 bars (or at least 10 if you prefer)
#      and compute the same features used in training (rolling average, etc.)
#    - Call model.predict() for the current bar and place trades accordingly.
# ----------------------------------------------------------------------------
class MLStrategy(bt.Strategy):
    params = (
        ('lookback', 20),  # how many bars we look back for computing features
    )
    
    def __init__(self):
        # Store reference to the trained model
        self.model = model
        
        # Keep aliases for data feeds for easy access
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_close = self.datas[0].close
        self.data_volume = self.datas[0].volume

    def next(self):
        # We need enough bars to compute features (lookback for average, etc.)
        if len(self) < self.p.lookback:
            return
        
        # Gather the last 'lookback' days of data to compute features
        lookback_range = range(-self.p.lookback, 0)
        
        closes = [self.data_close[i] for i in lookback_range]
        highs = [self.data_high[i] for i in lookback_range]
        lows  = [self.data_low[i] for i in lookback_range]
        opens = [self.data_open[i] for i in lookback_range]
        vols  = [self.data_volume[i] for i in lookback_range]
        
        # Current day's values (index=0 is the current bar)
        current_open = self.data_open[0]
        current_high = self.data_high[0]
        current_low  = self.data_low[0]
        current_close = self.data_close[0]
        current_volume = self.data_volume[0]
        
        # Compute a 10-day moving average for the *last 10 days* in this window
        # (You could adapt this to use the entire 'lookback' if you like.)
        recent_10_closes = closes[-10:]  # last 10 closes in the 20-day window
        ma10 = sum(recent_10_closes) / 10.0
        
        # Construct a single-row DataFrame with the same features used during training
        X_current = pd.DataFrame([{
            'open':   current_open,
            'high':   current_high,
            'low':    current_low,
            'close':  current_close,
            'volume': current_volume,
            'ma10':   ma10
        }])
        
        # Predict whether tomorrow's return is likely positive (1) or negative (0)
        prediction = self.model.predict(X_current)[0]
        
        # Simple trading logic: if we predict up and have no position, buy; else close
        if prediction == 1 and not self.position:
            self.buy(size=1)
        elif prediction == 0 and self.position:
            self.close()

# ----------------------------------------------------------------------------
# 6. Set up the Backtest on Test Data (2020 onward)
# ----------------------------------------------------------------------------
# We feed the test_df (raw) to Backtrader, so the strategy can compute features on-the-fly
cerebro = bt.Cerebro()
feed = bt.feeds.PandasData(dataname=test_df)
cerebro.adddata(feed)
cerebro.addstrategy(MLStrategy, lookback=20)

# Optional: Set initial capital
cerebro.broker.setcash(10000)

# Run the backtest
results = cerebro.run()

# Check final portfolio value
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Optionally plot the results
cerebro.plot()
