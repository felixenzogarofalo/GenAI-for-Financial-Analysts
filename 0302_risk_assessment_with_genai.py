import yfinance as yf
import backtrader as bt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import random

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
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])
    
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    
    return data

symbol = 'AAPL'
df = fetch_data(symbol, '2019-01-01', '2021-01-01')
train_df = df.loc[:'2019-12-31'].copy()
test_df = df.loc['2020-01-01':'2021-01-01'].copy()

# ----------------------------------------------------------------------------
# 2. Neural Network for Synthetic Data Generation
# ----------------------------------------------------------------------------

scaler = MinMaxScaler()
# 1. Prepare the data
scaled_data = scaler.fit_transform(train_df[['open', 'high', 'low', 'close', 'volume']])
train_X = np.array([scaled_data[i:i+10] for i in range(len(scaled_data)-10)])  # Shape (N, 10, 5)
train_Y = np.array([scaled_data[i+10] for i in range(len(scaled_data)-10)])    # Shape (N, 5)

train_X = torch.tensor(train_X, dtype=torch.float32)  # (N, 10, 5)
train_Y = torch.tensor(train_Y, dtype=torch.float32)  # (N, 5)

# 2. Flatten the 10x5 into 50 features
N = train_X.shape[0]
train_X = train_X.view(N, -1)  # Now shape is (N, 50)

# 3. Adjust input dimension to 50
input_dim = 50
hidden_dim = 20
output_dim = 5

# 4. Define the generator accordingly
class PriceGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PriceGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

generator = PriceGenerator(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    output = generator(train_X)
    loss = criterion(output, train_Y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# ----------------------------------------------------------------------------
# 3. Generate Synthetic Data Using the Trained Model
# ----------------------------------------------------------------------------
def generate_synthetic_data_nn(generator, num_scenarios=20, num_days=252, crisis=False):
    """
    Generate synthetic financial data using a trained neural network model.
    If crisis=True, simulate a market crash scenario.
    """
    synthetic_data = []
    
    with torch.no_grad():
        for _ in range(num_scenarios):
            prices = []
            seed = train_X[-1].view(1, -1, 5)
            random_black_swan_day = random.randint(0, num_days-50)
            for i in range(num_days):
                next_price = generator(seed.view(1, -1)) + torch.randn(5) * 0.02 # Add noise for variability
                if crisis and i == random_black_swan_day:
                    next_price *= np.random.uniform(0.7, 0.90)
                prices.append(next_price.squeeze(0).numpy())
                seed = torch.cat((seed[:, 1:, :], next_price.view(1, 1, 5)), dim=1)
            synthetic_data.append(scaler.inverse_transform(np.array(prices)))
    
    return synthetic_data

synthetic_scenarios = generate_synthetic_data_nn(generator, crisis=True)

scenario_results = []

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
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=5000, random_state=42)

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
        self.max_drawdown = max(self.drawdowns) if self.drawdowns else 0
        self.returns = (final_value - self.start_cash) / self.start_cash
        std_dev = np.std(self.drawdowns)
        self.sharpe_ratio = self.returns / std_dev if std_dev != 0 else np.nan
        self.var_95 = np.percentile(self.drawdowns, 5)
        self.expected_shortfall = np.mean([d for d in self.drawdowns if d <= self.var_95])
        
        print(f'Final Portfolio Value: {final_value:.2f}')
        print(f'Maximum Drawdown: {self.max_drawdown:.4f}')
        print(f'Sharpe Ratio: {self.sharpe_ratio:.4f}')
        print(f'Value at Risk (VaR 95%): {self.var_95:.4f}')
        print(f'Expected Shortfall (ES): {self.expected_shortfall:.4f}')


# ----------------------------------------------------------------------------
# 5. Run Backtesting on Synthetic Data
# ----------------------------------------------------------------------------
for i, scenario in enumerate(synthetic_scenarios):
    cerebro = bt.Cerebro()
    
    # Ensure the synthetic data has a datetime index
    synthetic_df = pd.DataFrame(scenario, columns=['open', 'high', 'low', 'close', 'volume'])
    synthetic_df['date'] = pd.date_range(start="2023-01-01", periods=len(scenario), freq='D')
    synthetic_df.set_index('date', inplace=True)
    
    # Convert it to a Backtrader-compatible data feed
    data = bt.feeds.PandasData(dataname=synthetic_df)
    
    cerebro.addstrategy(AIEnhancedStrategy)
    cerebro.adddata(data)
    cerebro.broker.set_cash(10000)
    results = cerebro.run()
    strategy_results = results[0]
    scenario_results.append({
            "Scenario": f"Scenario_{i+1}",
            "Final Portfolio Value": cerebro.broker.getvalue(),
            "Maximum Drawdown": strategy_results.max_drawdown,
            "Sharpe Ratio": strategy_results.sharpe_ratio,
            "Value at Risk (VaR 95%)": strategy_results.var_95,
            "Expected Shortfall (ES)": strategy_results.expected_shortfall
        })


# ----------------------------------------------------------------------------
# 6. Generate Synthetic Financial Report
# ----------------------------------------------------------------------------
def generate_synthetic_financial_report(output_file="synthetic_financial_report.xlsx"):
    """
    Export synthetic data and risk assessment metrics for Power BI visualization.
    """
    with pd.ExcelWriter(output_file) as writer:
        for i, scenario in enumerate(synthetic_scenarios):
            synthetic_df = pd.DataFrame(scenario, columns=['open', 'high', 'low', 'close', 'volume'])
            synthetic_df['date'] = pd.date_range(start="2023-01-01", periods=len(scenario), freq='D')
            synthetic_df.set_index('date', inplace=True)
            synthetic_df.to_excel(writer, sheet_name=f"Scenario_{i+1}", index=True)

        scenario_metrics_df = pd.DataFrame(scenario_results)
        scenario_metrics_df.to_excel(writer, sheet_name="Scenario Metrics", index=False)
    
    print(f"Synthetic Financial report saved to {output_file}")

generate_synthetic_financial_report()
