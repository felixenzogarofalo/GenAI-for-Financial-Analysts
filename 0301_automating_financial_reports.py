import yfinance as yf
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

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
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                         max_iter=1500, learning_rate_init=0.001, random_state=42)

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
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0

        # Data storage for reporting
        self.cash_history = []
        self.portfolio_value_history = []
        self.trade_log = []
        self.buy_signals = []
        self.sell_signals = []

    def log_trade(self, trade_type, price, size):
        """Log buy/sell trades."""
        self.trade_log.append({"Date": self.datetime.date(0), "Type": trade_type, "Price": price, "Size": size})

    def next(self):
        # Ensure we have enough data points before calculations
        if len(self) < self.p.lookback or len(self.dataclose) < 10:
            return  # Skip processing until we have enough data
        
        # Extract last 10 closing prices
        last_10_closes = [self.dataclose[i] for i in range(-10, 0)]
        
        # Check for NaN values before calculation
        if any(pd.isna(last_10_closes)):  # If any NaN values exist, replace with previous valid value
            last_10_closes = [x if not pd.isna(x) else last_10_closes[-1] for x in last_10_closes]

        # Compute statistics with error handling
        ma10 = np.mean(last_10_closes) if len(last_10_closes) == 10 else 0
        volatility = np.std(last_10_closes, ddof=1) if len(last_10_closes) == 10 and np.std(last_10_closes) != 0 else 0
        
        # Final safety measure: Convert NaNs to zero
        ma10 = np.nan_to_num(ma10, nan=0.0)
        volatility = np.nan_to_num(volatility, nan=0.0)

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
            self.log_trade("BUY", self.dataclose[0], 1)
            self.buy_signals.append({"Date": self.datetime.date(0), "Price": self.dataclose[0]})
        elif prediction == 0 and self.position:
            self.close()
            self.log_trade("SELL", self.dataclose[0], 1)
            self.sell_signals.append({"Date": self.datetime.date(0), "Price": self.dataclose[0]})

        # Calculate drawdown
        portfolio_value = self.broker.getvalue()
        self.high_watermark = max(self.high_watermark, portfolio_value)
        drawdown = (self.high_watermark - portfolio_value) / self.high_watermark
        self.drawdowns.append(drawdown)

        # Track portfolio values and cash
        self.cash_history.append({"Date": self.datetime.date(0), "Cash": self.broker.get_cash()})
        self.portfolio_value_history.append({"Date": self.datetime.date(0), "Value": self.broker.getvalue()})

    def stop(self):
        final_value = self.broker.getvalue()
        self.max_drawdown = max(self.drawdowns) if self.drawdowns else 0
        returns = (final_value - self.start_cash) / self.start_cash
        std_dev = np.std(self.drawdowns)
        self.sharpe_ratio = returns / std_dev if std_dev != 0 else np.nan

# ----------------------------------------------------------------------------
# 5. Running the Backtest
# ----------------------------------------------------------------------------
cerebro = bt.Cerebro()
cerebro.addstrategy(AIEnhancedStrategy, lookback=20)
cerebro.adddata(bt.feeds.PandasData(dataname=test_df))
cerebro.broker.set_cash(10000)
results = cerebro.run()
strategy = results[0]

def generate_ai_summary(trading_results):
    """
    Use Generative AI (Hugging Face Transformers) to create a financial report summary based on backtest results.

    Parameters:
    - trading_results: Dictionary containing key financial metrics.

    Returns:
    - AI-generated summary text.
    """
    # Construct a summary prompt using trading results
    summary_input = f"""
    Generate a professional financial summary based on the following trading results:
    
    - Final Portfolio Value: ${trading_results['Final Portfolio Value']:.2f}
    - Maximum Drawdown: {trading_results['Maximum Drawdown']:.2%}
    - Sharpe Ratio: {trading_results['Sharpe Ratio']:.2f}

    Explain whether the trading strategy was successful and highlight key risks.
    """
    
    # Load the Hugging Face - A 7B parameter GPT-like model
    # fine-tuned on a mix of publicly available, synthetic datasets
    summarizer = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha")
    
    # Generate summaryç
    summary = summarizer(summary_input, min_length=50, do_sample=False)
    
    # Return the summary text
    return summary[0]['generated_text']

# Generate the AI-powered summary
# Create a dictionary of results
results_dict = {
    "Final Portfolio Value": cerebro.broker.getvalue(),
    "Maximum Drawdown": strategy.max_drawdown,
    "Sharpe Ratio": strategy.sharpe_ratio,
}

# Now pass this dictionary to generate_ai_summary
ai_summary = generate_ai_summary(results_dict)
print(ai_summary)

# ----------------------------------------------------------------------------
# 6. Generating Automated Financial Reports for Power BI
# ----------------------------------------------------------------------------
def generate_financial_report(output_file="financial_report.xlsx"):
    """
    Generate an automated financial report for Power BI.

    This function exports:
    - Cash history
    - Portfolio value history
    - Trade history (Buy/Sell)
    - Stock price history with Open, High, Low, Close (OHLC)
    - Volume history
    - Buy/Sell signals
    - Summary with key metrics (Final Portfolio Value, Max Drawdown, Sharpe Ratio)
    """
    with pd.ExcelWriter(output_file) as writer:
        # Export Cash and Portfolio Value History
        pd.DataFrame(strategy.cash_history).to_excel(writer, sheet_name="Cash History", index=False)
        pd.DataFrame(strategy.portfolio_value_history).to_excel(writer, sheet_name="Portfolio Value", index=False)

        # Export Trade History
        pd.DataFrame(strategy.trade_log).to_excel(writer, sheet_name="Trades", index=False)

        # Export Stock Price History (OHLC + Volume) in a single table
        stock_data = test_df[['open', 'high', 'low', 'close', 'volume']].reset_index()
        stock_data.rename(columns={"Date": "Date"}, inplace=True)  # Ensure proper naming for Power BI
        stock_data.to_excel(writer, sheet_name="Stock Data (OHLC)", index=False)

        # Export Buy & Sell Signals
        pd.DataFrame(strategy.buy_signals).to_excel(writer, sheet_name="Buy Signals", index=False)
        pd.DataFrame(strategy.sell_signals).to_excel(writer, sheet_name="Sell Signals", index=False)

        # Export Summary of Key Metrics
        summary_data = {
            "Final Portfolio Value": cerebro.broker.getvalue(),
            "Maximum Drawdown": max(strategy.drawdowns) if strategy.drawdowns else 0,
            "Sharpe Ratio": strategy.sharpe_ratio if hasattr(strategy, 'sharpe_ratio') else "N/A",
            "AI Summary": ai_summary
        }
        pd.DataFrame([summary_data]).to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ Financial report saved to {output_file}")


generate_financial_report()
