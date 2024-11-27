import time
import json
import requests
from solana.rpc.api import Client
from solana.rpc.websocket_api import WebSocketClient
import pandas as pd
from datetime import datetime

class BlockchainConnector:
    def __init__(self, solana_rpc_url="https://api.mainnet-beta.solana.com"):
        """
        Initializes the BlockchainConnector to interact with the Solana blockchain.
        
        solana_rpc_url: str, optional (default='https://api.mainnet-beta.solana.com')
            The URL of the Solana RPC endpoint to connect to. This could be a public RPC or a custom one.
        """
        self.client = Client(solana_rpc_url)
        self.websocket_client = None  # Placeholder for WebSocket client
        self.transaction_cache = {}  # Cache to store previously seen transactions by signature
        self.transaction_data = []  # To store transaction data
        self.solana_rpc_url = solana_rpc_url
        print(f"BlockchainConnector initialized with Solana RPC: {self.solana_rpc_url}")
    
    def fetch_recent_transactions(self, limit=10):
        """
        Fetches the most recent transactions from the Solana blockchain.
        
        limit: int, optional (default=10)
            The number of recent transactions to fetch.
        
        Returns:
        --------
        transactions: list of dict
            A list of transaction data dictionaries.
        """
        try:
            # Fetch the most recent block
            recent_block = self.client.get_slot()
            print(f"Fetching transactions for the most recent block: {recent_block['result']}")
            
            # Get the block details (transactions)
            transactions = self.client.get_block(recent_block['result'], encoding="json")
            
            if 'result' in transactions and transactions['result']:
                return transactions['result']['transactions'][:limit]
            else:
                print("No transactions found in the block.")
                return []
        except Exception as e:
            print(f"Error fetching recent transactions: {e}")
            return []
    
    def process_transaction(self, tx_data):
        """
        Processes the transaction data to extract relevant details.
        
        tx_data: dict
            A dictionary containing transaction data.
        
        Returns:
        --------
        processed_data: dict
            A dictionary containing processed transaction data (amount, sender, receiver, etc.).
        """
        try:
            tx_details = tx_data['transaction']['message']
            processed_data = {
                'signature': tx_data['transaction']['signatures'][0],
                'slot': tx_data['slot'],
                'block_time': tx_data.get('blockTime', None),
                'amount': 0,  # Initialize with zero, we will calculate it later
                'sender': None,
                'receiver': None,
                'fee': tx_data['meta']['fee'] / 10**9  # Fee in SOL
            }
            
            # Extract instructions and transaction details
            for instruction in tx_details['instructions']:
                if 'parsed' in instruction:
                    parsed = instruction['parsed']
                    if parsed['type'] == 'transfer':
                        amount = int(parsed['info']['lamports']) / 10**9  # Lamports to SOL
                        sender = parsed['info']['source']
                        receiver = parsed['info']['destination']
                        processed_data['amount'] += amount
                        processed_data['sender'] = sender
                        processed_data['receiver'] = receiver
            
            return processed_data
        except KeyError as e:
            print(f"Error processing transaction: missing key {e}")
            return None
        except Exception as e:
            print(f"Unexpected error processing transaction: {e}")
            return None
    
    def store_transaction(self, tx_data):
        """
        Stores the transaction data in memory (or database if connected).
        
        tx_data: dict
            The processed transaction data.
        """
        if tx_data and tx_data['signature'] not in self.transaction_cache:
            # Store in the cache to avoid duplicate processing
            self.transaction_cache[tx_data['signature']] = tx_data
            self.transaction_data.append(tx_data)
            print(f"Stored transaction: {tx_data['signature']}")
        else:
            print(f"Transaction {tx_data['signature']} already processed or invalid.")
    
    def get_transaction_data(self):
        """
        Returns the list of all processed transaction data.
        
        Returns:
        --------
        list of dict
            A list of processed transaction data dictionaries.
        """
        return self.transaction_data
    
    def start_realtime_stream(self):
        """
        Starts the real-time transaction streaming using WebSockets.
        This listens to the Solana network for new transactions and processes them.
        """
        try:
            self.websocket_client = WebSocketClient(self.solana_rpc_url)
            
            # Define callback to process incoming transactions
            def on_transaction_received(transaction_data):
                processed_tx = self.process_transaction(transaction_data)
                self.store_transaction(processed_tx)
            
            # Subscribe to new transactions (all transactions)
            self.websocket_client.on('transaction', on_transaction_received)
            print("Real-time transaction stream started...")
            
            self.websocket_client.start()
        except Exception as e:
            print(f"Error starting WebSocket connection: {e}")
    
    def stop_realtime_stream(self):
        """
        Stops the real-time transaction streaming.
        """
        if self.websocket_client:
            self.websocket_client.stop()
            print("Real-time transaction stream stopped.")
    
    def poll_transactions(self, poll_interval=30, limit=10):
        """
        Polls the Solana blockchain for new transactions at regular intervals.
        
        poll_interval: int, optional (default=30)
            The time interval (in seconds) to wait between polling.
        limit: int, optional (default=10)
            The maximum number of transactions to fetch per poll.
        """
        try:
            while True:
                print(f"Polling for transactions every {poll_interval} seconds...")
                transactions = self.fetch_recent_transactions(limit=limit)
                
                if transactions:
                    for tx in transactions:
                        processed_tx = self.process_transaction(tx)
                        self.store_transaction(processed_tx)
                
                time.sleep(poll_interval)
        except Exception as e:
            print(f"Error while polling transactions: {e}")
    
    def fetch_transaction_by_signature(self, signature):
        """
        Fetches a specific transaction by its signature.
        
        signature: str
            The transaction signature (hash).
        
        Returns:
        --------
        dict or None
            The transaction data if found, otherwise None.
        """
        try:
            result = self.client.get_transaction(signature, encoding="json")
            if 'result' in result and result['result']:
                return result['result']
            else:
                print(f"Transaction {signature} not found.")
                return None
        except Exception as e:
            print(f"Error fetching transaction by signature {signature}: {e}")
            return None
    
    def get_transaction_by_slot(self, slot):
        """
        Fetches a transaction by its slot.
        
        slot: int
            The block slot number.
        
        Returns:
        --------
        dict or None
            The transaction data for the given slot, otherwise None.
        """
        try:
            result = self.client.get_block(slot, encoding="json")
            if 'result' in result and result['result']:
                return result['result']['transactions']
            else:
                print(f"No transactions found in slot {slot}.")
                return None
        except Exception as e:
            print(f"Error fetching transaction by slot {slot}: {e}")
            return None
    
    def visualize_transaction_data(self):
        """
        Visualizes the processed transaction data using basic charts.
        For example, plotting transaction amounts over time.
        """
        try:
            df = pd.DataFrame(self.transaction_data)
            
            # Plot total transaction amount over time
            df['block_time'] = pd.to_datetime(df['block_time'], unit='s')
            df.set_index('block_time', inplace=True)
            df['amount'].resample('D').sum().plot(kind='line', title='Transaction Amount Over Time')
            plt.xlabel('Date')
            plt.ylabel('Amount (SOL)')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error visualizing transaction data: {e}")