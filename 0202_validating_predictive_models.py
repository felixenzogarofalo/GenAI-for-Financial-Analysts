import yfinance as yf
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Fetch data
symbol = 'AAPL'
df = fetch_data(symbol, '2019-01-01', '2021-01-01')
train_df = df.loc[:'2019-12-31'].copy()
test_df = df.loc['2020-01-01':'2021-01-01'].copy()

# Feature Engineering
train_df['return'] = train_df['close'].pct_change()
train_df['ma10'] = train_df['close'].rolling(window=10).mean()
train_df.dropna(inplace=True)

X_train = train_df[['open', 'high', 'low', 'close', 'volume', 'ma10']]
y_train = (train_df['return'] > 0).astype(int)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on training set for ML metrics
y_train_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)

print(f'ML Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# ----------------------------------------------------------------------------
# 2. Backtrader Strategy: Online Predictions & Performance Metrics
# ----------------------------------------------------------------------------
class MLStrategy(bt.Strategy):
    params = (('lookback', 20),)
    
    def __init__(self):
        self.model = model
        self.dataclose = self.datas[0].close
        self.start_cash = self.broker.get_cash()
        self.high_watermark = self.start_cash
        self.drawdowns = []

    def next(self):
        if len(self) < self.p.lookback:
            return
        
        # Prepare current data for prediction
        ma10 = np.mean([self.dataclose[i] for i in range(-10, 0)])
        X_current = pd.DataFrame([{
            'open': self.datas[0].open[0],
            'high': self.datas[0].high[0],
            'low': self.datas[0].low[0],
            'close': self.datas[0].close[0],
            'volume': self.datas[0].volume[0],
            'ma10': ma10
        }])
        
        prediction = self.model.predict(X_current)[0]
        
        # Trading logic
        if prediction == 1 and not self.position:
            self.buy(size=1)
        elif prediction == 0 and self.position:
            self.close()
        
        # Calculate drawdown
        portfolio_value = self.broker.getvalue()
        self.high_watermark = max(self.high_watermark, portfolio_value)
        drawdown = (self.high_watermark - portfolio_value) / self.high_watermark
        self.drawdowns.append(drawdown)

    def stop(self):
        final_value = self.broker.getvalue()
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0
        returns = (final_value - self.start_cash) / self.start_cash
        std_dev = np.std(self.drawdowns)
        sharpe_ratio = returns / std_dev if std_dev != 0 else np.nan
        
        print(f'Final Portfolio Value: {final_value:.2f}')
        print(f'Maximum Drawdown: {max_drawdown:.4f}')
        print(f'Sharpe Ratio: {sharpe_ratio:.4f}')

# ----------------------------------------------------------------------------
# 3. Running the Backtest
# ----------------------------------------------------------------------------
cerebro = bt.Cerebro()
cerebro.addstrategy(MLStrategy, lookback=20)
cerebro.adddata(bt.feeds.PandasData(dataname=test_df))
cerebro.broker.set_cash(10000)
cerebro.run()
cerebro.plot()
