import yfinance as yf
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

# ----------------------------------------------------------------------------
# 2. Feature Engineering
# ----------------------------------------------------------------------------
train_df['return'] = train_df['close'].pct_change()
train_df['ma10'] = train_df['close'].rolling(window=10).mean()
train_df['volatility'] = train_df['close'].rolling(window=10).std()
train_df.dropna(inplace=True)

# Define features and labels
X_train = train_df[['open', 'high', 'low', 'close', 'volume', 'ma10', 'volatility']]
y_train = (train_df['return'] > 0).astype(int)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ----------------------------------------------------------------------------
# 3. Train Advanced Models
# ----------------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=500, random_state=42)

# Ensemble Model: Combines Random Forest, Gradient Boosting, and Neural Network
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('gb', gb_model),
    ('nn', nn_model)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# ----------------------------------------------------------------------------
# 4. Backtrader Strategy: AI-Powered Trading
# ----------------------------------------------------------------------------
class AIEnhancedStrategy(bt.Strategy):
    params = (('lookback', 20),)

    def __init__(self):
        self.model = ensemble_model
        self.scaler = scaler
        self.dataclose = self.datas[0].close
        self.start_cash = self.broker.get_cash()
        self.high_watermark = self.start_cash
        self.drawdowns = []

    def next(self):
        if len(self) < self.p.lookback:
            return

        # Compute features dynamically
        ma10 = np.mean([self.dataclose[i] for i in range(-10, 0)])
        volatility = np.std([self.dataclose[i] for i in range(-10, 0)])

        X_current = pd.DataFrame([{
            'open': self.datas[0].open[0],
            'high': self.datas[0].high[0],
            'low': self.datas[0].low[0],
            'close': self.datas[0].close[0],
            'volume': self.datas[0].volume[0],
            'ma10': ma10,
            'volatility': volatility
        }])

        # Scale input features
        X_current_scaled = self.scaler.transform(X_current)

        # Predict market direction
        prediction = self.model.predict(X_current_scaled)[0]

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
# 5. Running the Backtest
# ----------------------------------------------------------------------------
cerebro = bt.Cerebro()
cerebro.addstrategy(AIEnhancedStrategy, lookback=20)
cerebro.adddata(bt.feeds.PandasData(dataname=test_df))
cerebro.broker.set_cash(10000)
cerebro.run()
cerebro.plot()
