import os
import json

# Create necessary directories
os.makedirs('agents', exist_ok=True)
os.makedirs('db', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('utils', exist_ok=True)

# Create __init__.py files
with open('agents/__init__.py', 'w') as f:
    f.write('# Agents package\n')

# Create agent files
agent_files = {
    'agents/pricing_agent.py': '''import pandas as pd
import sqlite3
import numpy as np
from functools import lru_cache
import logging

# Get logger
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
        # Only select the columns we need
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
        return []''',
    
    'agents/demand_agent.py': '''import pandas as pd
import sqlite3
import numpy as np
from xgboost import XGBRegressor
from functools import lru_cache
import logging
import joblib
import os

# Get logger
logger = logging.getLogger('demand_agent')

# Cache connection to avoid repeated connection overhead
@lru_cache(maxsize=1)
def get_db_connection():
    try:
        return sqlite3.connect("db/retail.db")
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

# Cache the demand data to avoid redundant database queries
@lru_cache(maxsize=1)
def get_demand_data():
    try:
        conn = get_db_connection()
        # Only select the columns we need
        df = pd.read_sql(
            "SELECT Date, Price, Promotions, [Seasonality Factors], [External Factors], [Sales Quantity] FROM demand_forecasting", 
            conn
        )
        return df
    except Exception as e:
        logger.error(f"Error fetching demand data: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# Path for saving the trained model
MODEL_PATH = 'models/demand_model.joblib'

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Feature engineering function
def preprocess_data(df):
    """Preprocess the demand data for model training or prediction."""
    # Convert date to datetime once
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract time features efficiently
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Vectorized operations instead of apply
    df['PromotionFlag'] = np.where(df['Promotions'] == 'Yes', 1, 0)
    
    # Convert categorical variables to codes
    df['Seasonality'] = df['Seasonality Factors'].astype('category').cat.codes
    df['External'] = df['External Factors'].astype('category').cat.codes
    
    return df

# Train and save model
def train_demand_model(force_retrain=False):
    """Train the demand forecasting model and save it to disk."""
    # Check if model already exists and we're not forcing a retrain
    if os.path.exists(MODEL_PATH) and not force_retrain:
        logger.info(f"Using existing model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    
    try:
        # Get and preprocess data
        df = get_demand_data()
        df = preprocess_data(df)
        
        # Define features
        features = ['DayOfYear', 'Price', 'PromotionFlag', 'Seasonality', 'External']
        X = df[features]
        y = df['Sales Quantity']
        
        # Train model with optimized parameters
        model = XGBRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X, y)
        
        # Save model to disk
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model trained and saved to {MODEL_PATH}")
        
        return model
    except Exception as e:
        logger.error(f"Error training demand model: {e}")
        raise

def forecast_demand():
    """Forecast demand for the next day using the trained model."""
    try:
        # Get data and model
        df = get_demand_data()
        df = preprocess_data(df)
        
        # Load or train model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info("Loaded existing demand model")
        else:
            model = train_demand_model()
            logger.info("Trained new demand model")
        
        # Define features
        features = ['DayOfYear', 'Price', 'PromotionFlag', 'Seasonality', 'External']
        
        # Create next day prediction data
        next_day = pd.DataFrame([
            {
                'DayOfYear': df['DayOfYear'].max() + 1,
                'Price': df['Price'].mean(),
                'PromotionFlag': 1,  # Assuming promotion is active
                'Seasonality': 2,    # Using a default seasonality value
                'External': 1         # Using a default external factor value
            }
        ])
        
        # Make prediction
        prediction = model.predict(next_day[features])[0]
        result = round(float(prediction), 2)  # Ensure it's a native Python float
        
        logger.info(f"Demand forecast: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in demand forecasting: {e}")
        # Return a reasonable default in case of error
        return 0.0''',
    
    'agents/inventory_agent.py': '''import pandas as pd
import sqlite3
import numpy as np
from functools import lru_cache
import logging

# Get logger
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
        return []''',
    
    'agents/decision_agent.py': '''import logging
import json

# Get logger
logger = logging.getLogger('decision_agent')

def make_decision(demand, inventory_alerts, pricing_suggestions):
    """Generate a decision plan based on demand forecasts, inventory alerts, and pricing suggestions.
    
    Args:
        demand (float): Forecasted demand value
        inventory_alerts (list): List of inventory alerts
        pricing_suggestions (list): List of pricing suggestions
        
    Returns:
        str: A decision plan with reordering, pricing, and strategic recommendations
    """
    try:
        # Log the inputs for debugging
        logger.info(f"Making decision with demand: {demand}")
        logger.info(f"Inventory alerts count: {len(inventory_alerts)}")
        logger.info(f"Pricing suggestions count: {len(pricing_suggestions)}")
        
        # Format the data for better processing
        formatted_inventory = json.dumps(inventory_alerts[:5], indent=2) if inventory_alerts else "No alerts"
        formatted_pricing = json.dumps(pricing_suggestions[:5], indent=2) if pricing_suggestions else "No suggestions"
        
        # In Colab, we'll generate a simulated response instead of using an LLM
        
        # Generate reorder recommendations
        reorder_products = []
        for alert in inventory_alerts:
            reorder_products.append({
                'Product ID': alert['Product ID'],
                'Store ID': alert['Store ID'],
                'Quantity': alert['Order Quantity'],
                'Urgency': alert['Urgency']
            })
        
        # Generate price adjustment recommendations
        price_adjustments = []
        for item in pricing_suggestions:
            if item['Price'] != item['Suggested Price']:
                price_adjustments.append({
                    'Product ID': item['Product ID'],
                    'Store ID': item['Store ID'],
                    'Current Price': item['Price'],
                    'Suggested Price': item['Suggested Price'],
                    'Change %': item['Price Change %']
                })
        
        # Generate strategic recommendations based on data
        strategies = []
        if demand > 100:
            strategies.append("Increase inventory levels for high-demand products")
        else:
            strategies.append("Maintain current inventory levels")
            
        if len(price_adjustments) > 5:
            strategies.append("Implement price adjustments gradually to monitor market response")
        
        if any(alert['Urgency'] == 'High' for alert in inventory_alerts):
            strategies.append("Expedite shipping for high-urgency items to prevent stockouts")
        
        # Format the decision plan
        decision_plan = f"""
# Inventory Optimization Decision Plan

## 1. Products to Reorder

{json.dumps(reorder_products[:5], indent=2) if reorder_products else "No products need reordering at this time."}
{f"... and {len(reorder_products) - 5} more" if len(reorder_products) > 5 else ""}

## 2. Price Adjustments

{json.dumps(price_adjustments[:5], indent=2) if price_adjustments else "No price adjustments recommended at this time."}
{f"... and {len(price_adjustments) - 5} more" if len(price_adjustments) > 5 else ""}

## 3. Strategic Recommendations

{"".join(f"- {strategy}\\n" for strategy in strategies)}

## 4. Demand Forecast

The forecasted demand is {demand} units. Plan inventory accordingly.
"""
        
        logger.info("Decision plan generated successfully")
        return decision_plan
        
    except Exception as e:
        logger.error(f"Error in decision making: {e}")
        return "Error: Unable to generate decision plan. Please check the logs for details."''',
    
    'agents/customer_agent.py': '''import random
import numpy as np
import logging
from functools import lru_cache

# Get logger
logger = logging.getLogger('customer_agent')

# Price elasticity model - cached for performance
@lru_cache(maxsize=100)
def calculate_purchase_probability(price, customer_segment=1):
    """Calculate purchase probability based on price and customer segment.
    
    Args:
        price (float): Product price
        customer_segment (int): Customer segment (1=price sensitive, 2=quality focused, 3=luxury)
        
    Returns:
        float: Probability of purchase between 0 and 1
    """
    # Base probability by segment
    base_probs = {1: 0.8, 2: 0.6, 3: 0.4}
    base_prob = base_probs.get(customer_segment, 0.6)
    
    # Price thresholds by segment
    thresholds = {1: 30, 2: 50, 3: 100}
    threshold = thresholds.get(customer_segment, 40)
    
    # Calculate probability with sigmoid function for smoother transition
    k = 0.1  # Steepness of the curve
    x0 = threshold  # Midpoint of the curve
    prob = base_prob / (1 + np.exp(k * (price - x0)))
    
    return min(max(prob, 0.1), 0.9)  # Clamp between 0.1 and 0.9

def customer_response(price, stock, customer_segment=1, batch_size=1):
    """Simulate customer purchase response based on price and stock availability.
    
    Args:
        price (float): Product price
        stock (int): Available stock
        customer_segment (int, optional): Customer segment type. Defaults to 1.
        batch_size (int, optional): Number of customers to simulate. Defaults to 1.
        
    Returns:
        str or dict: Purchase decision(s)
    """
    try:
        # No stock means no purchase
        if stock <= 0:
            logger.info(f"No purchase possible - out of stock (price: ${price})")
            return "No purchase" if batch_size == 1 else {"Buy": 0, "No purchase": batch_size}
        
        # Calculate purchase probability
        prob = calculate_purchase_probability(price, customer_segment)
        
        if batch_size == 1:
            # Single customer simulation
            decision = "Buy" if random.random() < prob else "No purchase"
            logger.debug(f"Customer decision: {decision} (price: ${price}, probability: {prob:.2f})")
            return decision
        else:
            # Batch simulation - more efficient for multiple customers
            # Generate random numbers for all customers at once
            random_values = np.random.random(batch_size)
            purchases = np.sum(random_values < prob)
            
            result = {"Buy": int(purchases), "No purchase": batch_size - int(purchases)}
            logger.info(f"Batch simulation: {result} (price: ${price}, probability: {prob:.2f})")
            return result
            
    except Exception as e:
        logger.error(f"Error in customer response simulation: {e}")
        return "Error" if batch_size == 1 else {"Error": batch_size}'''
}

# Create utility files
utility_files = {
    'utils/init_db.py': '''import pandas as pd
import sqlite3
import os
import numpy as np
import random
from datetime import datetime, timedelta

# Create db folder if it doesn't exist
os.makedirs("db", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Connect to database
conn = sqlite3.connect("db/retail.db")

# Generate demand forecasting data
def generate_demand_data(num_records=100):
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_records)]
    
    data = {
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Product ID': np.random.randint(1000, 10000, num_records),
        'Store ID': np.random.randint(1, 100, num_records),
        'Price': np.round(np.random.uniform(5, 100, num_records), 2),
        'Promotions': np.random.choice(['Yes', 'No'], num_records, p=[0.3, 0.7]),
        'Seasonality Factors': np.random.choice(['High', 'Medium', 'Low'], num_records),
        'External Factors': np.random.choice(['Holiday', 'Normal', 'Weather Event'], num_records),
        'Sales Quantity': np.random.randint(10, 500, num_records)
    }
    
    return pd.DataFrame(data)

# Generate inventory monitoring data
def generate_inventory_data(num_records=100):
    data = {
        'Product ID': np.random.randint(1000, 10000, num_records),
        'Store ID': np.random.randint(1, 100, num_records),
        'Stock Levels': np.random.randint(0, 500, num_records),
        'Reorder Point': np.random.randint(10, 100, num_records),
        'Safety Stock': np.random.randint(5, 50, num_records),
        'Supplier Lead Time (days)': np.random.randint(1, 21, num_records),
        'Storage Cost': np.round(np.random.uniform(1, 10, num_records), 2)
    }
    
    return pd.DataFrame(data)

# Generate pricing optimization data
def generate_pricing_data(num_records=100):
    data = {
        'Product ID': np.random.randint(1000, 10000, num_records),
        'Store ID': np.random.randint(1, 100, num_records),
        'Price': np.round(np.random.uniform(5, 100, num_records), 2),
        'Competitor Prices': np.round(np.random.uniform(5, 100, num_records), 2),
        'Discounts': np.round(np.random.uniform(0, 50, num_records), 2),
        'Sales Volume': np.random.randint(0, 500, num_records),
        'Customer Reviews': np.random.randint(1, 6, num_records),
        'Return Rate (%)': np.round(np.random.uniform(0, 20, num_records), 2),
        'Storage Cost': np.round(np.random.uniform(1, 10, num_records), 2),
        'Elasticity Index': np.round(np.random.uniform(0.5, 2.5, num_records), 2)
    }
    
    return pd.DataFrame(data)

# Generate data
df_demand = generate_demand_data(100)
df_inventory = generate_inventory_data(100)
df_pricing = generate_pricing_data(100)

# Save to CSV (for reference)
df_demand.to_csv("data/demand_forecasting.csv", index=False)
df_inventory.to_csv("data/inventory_monitoring.csv", index=False)
df_pricing.to_csv("data/pricing_optimization.csv", index=False)

# Save to database
df_demand.to_sql("demand_forecasting", conn, if_exists="replace", index=False)
df_inventory.to_sql("inventory_monitoring", conn, if_exists="replace", index=False)
df_pricing.to_sql("pricing_optimization", conn, if_exists="replace", index=False)

conn.close()

print("Database initialized with sample data.")'''
}

# Create requirements.txt
requirements = '''streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
joblib>=1.3.0
plotly>=5.18.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pyngrok>=6.0.0
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements)

# Create README.md
readme = '''# Retail Inventory Optimization - Multi-Agent System (Colab Version)

This is a Colab-compatible version of the Retail Inventory Optimization multi-agent system.

## How to Run

1. Open the `retail_inventory_optimization.ipynb` notebook in Google Colab
2. Run all cells in order
3. Click on the ngrok URL to access the Streamlit app

## System Architecture

The system consists of multiple specialized agents:

1. **Demand Agent**: Forecasts future product demand using XGBoost
2. **Inventory Agent**: Monitors inventory levels and generates alerts for products that need reordering
3. **Pricing Agent**: Optimizes product pricing based on elasticity and competitor prices
4. **Customer Agent**: Simulates customer purchase behavior based on price and stock availability
5. **Decision Agent**: Coordinates between other agents to make strategic inventory decisions

## Performance Optimizations

The system has been optimized for speed and efficiency:

- Added caching with `@lru_cache` to avoid redundant database queries and calculations
- Implemented proper error handling and logging throughout the system
- Used vectorized operations instead of slow `apply()` functions
- Added parallel processing for model training
- Implemented batch processing for customer simulations
'''

with open('README.md', 'w') as f:
    f.write(readme)

# Write all files
for file_path, content in agent_files.items():
    with open(file_path, 'w') as f:
        f.write(content)

for file_path, content in utility_files.items():
    with open(file_path, 'w') as f:
        f.write(content)

print("Setup complete. All files have been created.")
