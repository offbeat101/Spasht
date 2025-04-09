import pandas as pd
import sqlite3
import numpy as np
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('inventory_agent')

# Cache connection to avoid repeated connection overhead
@lru_cache(maxsize=1)
def get_db_connection():
    try:
        return sqlite3.connect("db/retail.db")
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

# Cache the inventory data to avoid redundant database queries
@lru_cache(maxsize=1)
def get_inventory_data():
    try:
        conn = get_db_connection()
        # Only select the columns we need
        df = pd.read_sql(
            "SELECT [Product ID], [Store ID], [Stock Levels], [Supplier Lead Time (days)], [Reorder Point], [Safety Stock] FROM inventory_monitoring", 
            conn
        )
        return df
    except Exception as e:
        logger.error(f"Error fetching inventory data: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def monitor_inventory(daily_sales_rate=10):
    """Monitor inventory levels and generate alerts for products that need reordering.
    
    Args:
        daily_sales_rate (int, optional): Assumed daily sales rate. Defaults to 10.
        
    Returns:
        list: A list of dictionaries containing alerts for products that need reordering.
    """
    try:
        # Get inventory data
        df = get_inventory_data()
        
        # Calculate projected days of inventory using vectorized operations
        df['Projected Days'] = df['Stock Levels'] / daily_sales_rate
        
        # Create mask for items that need reordering
        reorder_mask = df['Projected Days'] < df['Supplier Lead Time (days)']
        
        # Filter dataframe using the mask
        reorder_df = df[reorder_mask].copy()
        
        # Calculate urgency level based on how soon we'll run out
        reorder_df['Urgency'] = np.where(
            reorder_df['Projected Days'] < reorder_df['Safety Stock'] / daily_sales_rate,
            'High',
            'Medium'
        )
        
        # Calculate recommended order quantity
        reorder_df['Order Quantity'] = np.ceil(
            (reorder_df['Supplier Lead Time (days)'] * daily_sales_rate * 1.5) - reorder_df['Stock Levels']
        ).astype(int)
        
        # Generate alerts using vectorized operations
        if len(reorder_df) > 0:
            alerts = reorder_df[['Product ID', 'Store ID', 'Supplier Lead Time (days)', 'Urgency', 'Order Quantity']].rename(
                columns={'Supplier Lead Time (days)': 'ETA'}
            ).assign(Action='Reorder').to_dict(orient='records')
        else:
            alerts = []
            
        logger.info(f"Generated {len(alerts)} inventory alerts")
        return alerts
        
    except Exception as e:
        logger.error(f"Error in inventory monitoring: {e}")
        # Return empty list in case of error to avoid breaking the application
        return []
