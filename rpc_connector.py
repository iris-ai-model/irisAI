import json
import requests
from time import sleep
from typing import Optional, Dict, Any

class RPCConnector:
    def __init__(self, solana_rpc_url="https://api.mainnet-beta.solana.com"):
        """
        Initializes the RPCConnector to interact with the Solana blockchain via the RPC API.
        
        solana_rpc_url: str, optional (default='https://api.mainnet-beta.solana.com')
            The URL of the Solana RPC endpoint to connect to. This could be a public RPC or a custom one.
        """
        self.solana_rpc_url = solana_rpc_url
        self.headers = {
            "Content-Type": "application/json",
        }
        print(f"RPCConnector initialized with Solana RPC URL: {self.solana_rpc_url}")
    
    def _send_request(self, method: str, params: Optional[list] = None) -> Dict[str, Any]:
        """
        Sends a request to the Solana RPC endpoint with the given method and parameters.
        
        method: str
            The RPC method to call (e.g., "getBlock", "getTransaction").
        
        params: list, optional
            A list of parameters for the RPC method (default is None).
        
        Returns:
        --------
        response: dict
            The response from the Solana RPC API.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or []
        }
        
        try:
            response = requests.post(self.solana_rpc_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()  # Will raise an HTTPError for bad responses
            result = response.json()
            
            if "error" in result:
                raise ValueError(f"Error from Solana RPC: {result['error']}")
            
            return result['result']
        
        except requests.exceptions.RequestException as e:
            print(f"RequestException occurred: {e}")
            return {}
        
        except ValueError as e:
            print(f"Error parsing RPC response: {e}")
            return {}
    
    def get_slot(self) -> Optional[int]:
        """
        Gets the current slot of the Solana blockchain.
        
        Returns:
        --------
        slot: int or None
            The current slot number if successful, otherwise None.
        """
        try:
            result = self._send_request("getSlot")
            return result
        except Exception as e:
            print(f"Error fetching slot: {e}")
            return None
    
    def get_block(self, slot: int) -> Optional[Dict[str, Any]]:
        """
        Fetches block data by slot.
        
        slot: int
            The slot number of the block to retrieve.
        
        Returns:
        --------
        block_data: dict or None
            The block data for the given slot, or None if an error occurs.
        """
        try:
            result = self._send_request("getBlock", [slot, {"encoding": "json"}])
            return result
        except Exception as e:
            print(f"Error fetching block data: {e}")
            return None
    
    def get_transaction_by_signature(self, signature: str) -> Optional[Dict[str, Any]]:
        """
        Fetches a transaction by its signature.
        
        signature: str
            The transaction signature to look up.
        
        Returns:
        --------
        transaction_data: dict or None
            The transaction data if found, or None if the transaction does not exist.
        """
        try:
            result = self._send_request("getTransaction", [signature, {"encoding": "json"}])
            return result
        except Exception as e:
            print(f"Error fetching transaction by signature {signature}: {e}")
            return None
    
    def get_account_info(self, pubkey: str) -> Optional[Dict[str, Any]]:
        """
        Fetches account information for a given public key.
        
        pubkey: str
            The public key of the account.
        
        Returns:
        --------
        account_info: dict or None
            The account information for the provided public key, or None if not found.
        """
        try:
            result = self._send_request("getAccountInfo", [pubkey, {"encoding": "json"}])
            return result
        except Exception as e:
            print(f"Error fetching account info for {pubkey}: {e}")
            return None
    
    def get_multiple_accounts_info(self, pubkeys: list) -> Optional[Dict[str, Any]]:
        """
        Fetches account information for multiple public keys.
        
        pubkeys: list of str
            A list of public keys for which to fetch account information.
        
        Returns:
        --------
        accounts_info: dict or None
            A dictionary containing account information for each public key, or None if an error occurs.
        """
        try:
            result = self._send_request("getMultipleAccounts", [pubkeys, {"encoding": "json"}])
            return result
        except Exception as e:
            print(f"Error fetching multiple accounts info: {e}")
            return None
    
    def get_epoch_info(self) -> Optional[Dict[str, Any]]:
        """
        Fetches the current epoch information.
        
        Returns:
        --------
        epoch_info: dict or None
            The epoch information if successful, otherwise None.
        """
        try:
            result = self._send_request("getEpochInfo")
            return result
        except Exception as e:
            print(f"Error fetching epoch info: {e}")
            return None
    
    def get_recent_performance_samples(self, limit: int = 10) -> Optional[Dict[str, Any]]:
        """
        Fetches recent performance samples for the Solana network.
        
        limit: int, optional (default=10)
            The number of performance samples to fetch.
        
        Returns:
        --------
        performance_samples: dict or None
            The performance sample data if successful, otherwise None.
        """
        try:
            result = self._send_request("getRecentPerformanceSamples", [limit])
            return result
        except Exception as e:
            print(f"Error fetching recent performance samples: {e}")
            return None
    
    def get_fees(self) -> Optional[Dict[str, Any]]:
        """
        Fetches the current fee structure for the Solana network.
        
        Returns:
        --------
        fees: dict or None
            The current fee structure data if successful, otherwise None.
        """
        try:
            result = self._send_request("getFees")
            return result
        except Exception as e:
            print(f"Error fetching fee data: {e}")
            return None
    
    def wait_for_slot(self, target_slot: int, timeout: int = 60) -> bool:
        """
        Waits for the blockchain to reach a specific slot.
        
        target_slot: int
            The target slot to wait for.
        
        timeout: int, optional (default=60)
            The number of seconds to wait before timing out.
        
        Returns:
        --------
        success: bool
            True if the slot is reached, False if the timeout is exceeded.
        """
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                current_slot = self.get_slot()
                if current_slot is None:
                    return False
                if current_slot >= target_slot:
                    print(f"Reached target slot: {current_slot}")
                    return True
                sleep(2)  # Poll every 2 seconds
            print("Timeout exceeded while waiting for slot.")
            return False
        except Exception as e:
            print(f"Error waiting for slot: {e}")
            return False
    
    def get_block_time(self, slot: int) -> Optional[int]:
        """
        Fetches the block time for a given slot.
        
        slot: int
            The slot number to retrieve block time for.
        
        Returns:
        --------
        block_time: int or None
            The block time as a Unix timestamp, or None if an error occurs.
        """
        try:
            result = self._send_request("getBlockTime", [slot])
            return result
        except Exception as e:
            print(f"Error fetching block time for slot {slot}: {e}")
            return None