# Overview

## Project Purpose

The Iris AI project is designed to facilitate interaction with the Solana blockchain, providing a Python-based framework for:

- Fetching transactions and blocks from the Solana blockchain.
- Interfacing with the Solana RPC API.
- Analyzing and processing Solana transaction data.
- Visualizing blockchain activity.

This project provides the tools for developers to integrate Solana blockchain data into their own applications, perform on-chain analysis, or build innovative solutions using the Solana network.

## Core Components

### BlockchainConnector
The `BlockchainConnector` class is designed to interface with the Solana blockchain directly, fetching data about recent transactions, processing blocks, and interacting with the blockchain state.

### RPCConnector
The `RPCConnector` class interacts with the Solana RPC API, enabling users to query blockchain data such as account info, transaction history, block information, and more. The RPCConnector helps extract data for analysis or other use cases.

### Data Visualization (Optional)
In the future, we may include a data visualization module that allows for graphical representation of Solana blockchain data, such as transaction trends, block throughput, and account balances.

---

## Architecture

The project follows a modular structure with clearly defined components:

1. **RPC Layer** (`rpc_connector.py`): Handles communication with Solana's RPC endpoints.
2. **Blockchain Layer** (`blockchain_connector.py`): Fetches and processes transactions, blocks, and other blockchain data.
3. **Utilities** (`utils.py`): Helper functions for data processing and formatting.
4. **Tests**: Unit tests and integration tests for each component.

The goal of this modular approach is to ensure maintainability, scalability, and ease of integration into other Python-based projects.

---

## Technologies

- **Solana Python SDK**: The core library used to interface with the Solana blockchain.
- **Requests**: Used for making HTTP requests to the Solana RPC API.
- **pytest**: Testing framework for validating the project's functionality.
- **Matplotlib (Optional)**: For visualizing blockchain data, if implemented.
- **pandas (Optional)**: For data manipulation and analysis (if needed for large datasets).

---

## License

This project is licensed under the MIT License.