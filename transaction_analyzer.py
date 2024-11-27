import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
import datetime

class IrisTransactionAnalyzer:
    def __init__(self):
        """Initializes the IrisTransactionAnalyzer to process transaction data."""
        self.transaction_data = []
        print("IrisTransactionAnalyzer initialized.")
        
    def process_transaction(self, tx_data):
        """Processes a single transaction's data."""
        try:
            # Extract relevant data fields from the transaction
            processed_data = {
                'amount': tx_data.get('amount', 0),
                'fee': tx_data.get('fee', 0),
                'sender': tx_data.get('sender', ''),
                'receiver': tx_data.get('receiver', ''),
                'timestamp': tx_data.get('timestamp', 0),
                'signature': tx_data.get('signature', ''),
                'block_time': tx_data.get('blockTime', 0),  # Assuming blockTime is available
                'slot': tx_data.get('slot', 0),  # Transaction slot (unique block identifier)
            }
            self.transaction_data.append(processed_data)
        except Exception as e:
            print(f"Error processing transaction: {e}")
    
    def get_processed_data(self):
        """Returns the processed transaction data as a pandas DataFrame."""
        return pd.DataFrame(self.transaction_data)
    
    def clean_data(self):
        """Cleans the transaction data by filling missing values and removing duplicates."""
        df = self.get_processed_data()
        # Remove duplicates
        df = df.drop_duplicates(subset='signature', keep='last')
        # Fill missing values with 0
        df = df.fillna(0)
        return df

    def basic_statistics(self):
        """Returns basic statistics about the transaction amounts and fees."""
        df = self.clean_data()
        stats = {
            'Total Transactions': len(df),
            'Total Amount Transacted': df['amount'].sum(),
            'Average Transaction Amount': df['amount'].mean(),
            'Average Fee': df['fee'].mean(),
            'Max Transaction Amount': df['amount'].max(),
            'Min Transaction Amount': df['amount'].min(),
            'Std Dev of Transaction Amount': df['amount'].std(),
        }
        return stats
    
    def detect_outliers(self, threshold=3):
        """Detects outliers in transaction amounts and fees using Z-score."""
        df = self.clean_data()
        amount_zscore = zscore(df['amount'])
        fee_zscore = zscore(df['fee'])
        
        # Transactions with Z-scores greater than the threshold are considered outliers
        amount_outliers = df[amount_zscore > threshold]
        fee_outliers = df[fee_zscore > threshold]
        
        print(f"Outliers detected in transaction amounts: {len(amount_outliers)} transactions")
        print(f"Outliers detected in transaction fees: {len(fee_outliers)} transactions")
        
        return amount_outliers, fee_outliers
    
    def plot_transaction_histogram(self):
        """Plots a histogram of transaction amounts."""
        df = self.clean_data()
        amounts = df['amount'].values
        plt.figure(figsize=(10, 6))
        plt.hist(amounts, bins=50, alpha=0.75, color='blue')
        plt.title('Transaction Amounts Distribution (Iris)')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_fee_distribution(self):
        """Plots a distribution of transaction fees."""
        df = self.clean_data()
        fees = df['fee'].values
        plt.figure(figsize=(10, 6))
        plt.hist(fees, bins=50, alpha=0.75, color='green')
        plt.title('Transaction Fee Distribution (Iris)')
        plt.xlabel('Fee')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_transaction_trends(self):
        """Plots trends of transaction amounts and fees over time (based on timestamp)."""
        df = self.clean_data()
        
        # Convert 'timestamp' into a more usable form, such as time-based intervals (e.g., days)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        # Plot transaction amounts over time
        plt.figure(figsize=(10, 6))
        df['amount'].resample('D').sum().plot(label='Transaction Amounts', color='blue')
        df['fee'].resample('D').sum().plot(label='Transaction Fees', color='green')
        plt.title('Transaction Amounts and Fees Trend (Iris)')
        plt.xlabel('Date')
        plt.ylabel('Amount/Fee')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_fee_vs_amount(self):
        """Plots a scatter plot of transaction fees versus transaction amounts."""
        df = self.clean_data()
        plt.figure(figsize=(10, 6))
        plt.scatter(df['amount'], df['fee'], alpha=0.5, color='purple')
        plt.title('Transaction Fee vs Amount (Iris)')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Transaction Fee')
        plt.grid(True)
        plt.show()

    def plot_transaction_volatility(self, window=7):
        """Plots the rolling volatility of transaction amounts."""
        df = self.clean_data()
        df['amount_rolling_std'] = df['amount'].rolling(window=window).std()
        plt.figure(figsize=(10, 6))
        df['amount_rolling_std'].plot(label=f'Rolling {window}-Day Volatility', color='orange')
        plt.title('Transaction Amount Volatility (Iris)')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Standard Deviation)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def transaction_amount_by_slot(self):
        """Aggregates and plots transaction amounts by blockchain slot."""
        df = self.clean_data()
        slot_data = df.groupby('slot')['amount'].sum()
        plt.figure(figsize=(10, 6))
        slot_data.plot(kind='bar', color='cyan')
        plt.title('Total Transaction Amount by Slot (Iris)')
        plt.xlabel('Slot')
        plt.ylabel('Total Amount Transacted')
        plt.grid(True)
        plt.show()

    # NEW FEATURE: Forecasting Future Transaction Amounts
    def forecast_transaction_amounts(self, periods=30):
        """Uses Holt-Winters Exponential Smoothing to forecast transaction amounts."""
        df = self.clean_data()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        # Resample daily and sum the transaction amounts
        daily_data = df['amount'].resample('D').sum()

        # Fit the model
        model = ExponentialSmoothing(daily_data, trend='add', seasonal=None, damped_trend=True)
        model_fit = model.fit()

        # Forecast future transaction amounts
        forecast = model_fit.forecast(periods)

        # Plot the forecast
        plt.figure(figsize=(10, 6))
        plt.plot(daily_data, label='Historical Data', color='blue')
        plt.plot(forecast, label='Forecast', color='red', linestyle='dashed')
        plt.title('Transaction Amount Forecast (Iris)')
        plt.xlabel('Date')
        plt.ylabel('Transaction Amount')
        plt.legend()
        plt.grid(True)
        plt.show()

    # NEW FEATURE: Correlation Matrix
    def plot_correlation_matrix(self):
        """Generates a heatmap for the correlation matrix of transaction features."""
        df = self.clean_data()
        correlation_matrix = df[['amount', 'fee', 'block_time', 'slot']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix of Transaction Features (Iris)')
        plt.show()

    # NEW FEATURE: KMeans Clustering of Transactions
    def cluster_transactions(self, num_clusters=3):
        """Clusters transactions using KMeans clustering based on amount and fee."""
        df = self.clean_data()
        
        # Standardize the data (important for clustering)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[['amount', 'fee']])

        # Fit KMeans model
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_data)

        # Plot clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='amount', y='fee', hue='cluster', palette='viridis', s=100, alpha=0.6)
        plt.title(f'KMeans Clustering of Transactions (Iris) - {num_clusters} Clusters')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Transaction Fee')
        plt.grid(True)
        plt.show()

    # NEW FEATURE: Heatmap of Transaction Density Over Time
    def plot_transaction_heatmap(self, time_interval='D'):
        """Plots a heatmap of transaction density over time."""
        df = self.clean_data()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['week'] = df['timestamp'].dt.week
        df['month'] = df['timestamp'].dt.month

        # Group by time interval
        time_data = df.groupby([time_interval, 'hour']).size().unstack(fill_value=0)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(time_data, cmap='YlGnBu', annot=True, fmt='d', linewidths=0.5)
        plt.title(f'Transaction Heatmap (Iris) - {time_interval} Interval')
        plt.ylabel('Time Interval')
        plt.xlabel('Hour of Day')
        plt.show()
