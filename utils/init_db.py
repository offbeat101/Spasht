import pandas as pd
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

print("Database initialized with sample data.")
