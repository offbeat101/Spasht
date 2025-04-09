import pandas as pd
import sqlite3
import numpy as np
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pricing_agent')

# Cache connection to avoid repeated connection overhead
@lru_cache(maxsize=1)
def get_db_connection():
    try:
        return sqlite3.connect("db/retail.db")
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

# Cache the pricing data to avoid redundant database queries
@lru_cache(maxsize=1)
def get_pricing_data():
    try:
        conn = get_db_connection()
        # Only select the columns we actually need
        df = pd.read_sql(
            "SELECT [Product ID], [Store ID], Price, [Competitor Prices], [Elasticity Index] FROM pricing_optimization", 
            conn
        )
        return df
    except Exception as e:
        logger.error(f"Error fetching pricing data: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def optimize_pricing():
    """Optimize product pricing based on elasticity and competitor prices.
    
    Returns:
        list: A list of dictionaries containing product pricing information and suggestions.
    """
    try:
        # Get pricing data
        df = get_pricing_data()
        
        # Create mask for high elasticity products (vectorized operation)
        high_elasticity_mask = df['Elasticity Index'] > 1.5
        
        # Initialize suggested price column with current prices
        df['Suggested Price'] = df['Price']
        
        # Calculate potential price reductions (vectorized)
        price_reduction = df.loc[high_elasticity_mask, 'Price'] * 0.9  # 10% reduction
        competitor_based_price = df.loc[high_elasticity_mask, 'Competitor Prices'] * 0.95
        
        # Use numpy's maximum function for vectorized comparison
        df.loc[high_elasticity_mask, 'Suggested Price'] = np.maximum(price_reduction, competitor_based_price)
        
        # Round prices to 2 decimal places for currency
        df['Suggested Price'] = df['Suggested Price'].round(2)
        
        # Calculate potential revenue impact
        df['Price Change'] = df['Suggested Price'] - df['Price']
        df['Price Change %'] = (df['Price Change'] / df['Price'] * 100).round(2)
        
        # Only return necessary columns in the result
        result_columns = ['Product ID', 'Store ID', 'Price', 'Suggested Price', 'Elasticity Index', 'Price Change %']
        
        # Convert to records for API compatibility
        result = df[result_columns].to_dict(orient='records')
        
        logger.info(f"Pricing optimization completed for {len(df)} products")
        return result
        
    except Exception as e:
        logger.error(f"Error in price optimization: {e}")
        # Return empty list in case of error to avoid breaking the application
        return []
