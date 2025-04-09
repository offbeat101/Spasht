import pandas as pd
import sqlite3
import numpy as np
from xgboost import XGBRegressor
from functools import lru_cache
import logging
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            "SELECT Date, Price, Promotions, [Seasonality Factors], [External Factors], [Sales Quantity], [Product ID], [Store ID] FROM demand_forecasting", 
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
SCALER_PATH = 'models/demand_scaler.joblib'
METRICS_PATH = 'models/demand_model_metrics.json'

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Feature engineering function
def preprocess_data(df):
    """Preprocess the demand data for model training or prediction."""
    # Convert date to datetime once
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract time features efficiently
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Vectorized operations instead of apply
    df['PromotionFlag'] = np.where(df['Promotions'] == 'Yes', 1, 0)
    
    # Convert categorical variables to codes
    df['Seasonality'] = df['Seasonality Factors'].astype('category').cat.codes
    df['External'] = df['External Factors'].astype('category').cat.codes
    
    # Create lag features if we have enough data
    if len(df) > 10:
        try:
            # Sort by date
            df = df.sort_values('Date')
            # Group by product and store
            for col in ['Sales Quantity', 'Price']:
                # Create lag features
                df[f'{col}_lag1'] = df.groupby(['Product ID', 'Store ID'])[col].shift(1)
                df[f'{col}_lag7'] = df.groupby(['Product ID', 'Store ID'])[col].shift(7)
                # Create rolling mean features
                df[f'{col}_rolling_mean_7'] = df.groupby(['Product ID', 'Store ID'])[col].transform(
                    lambda x: x.rolling(7, min_periods=1).mean())
            
            # Fill NaN values instead of dropping rows
            df = df.fillna(method='bfill').fillna(method='ffill')
            logger.info(f"Created lag features. New shape: {df.shape}")
        except Exception as e:
            logger.warning(f"Could not create lag features: {e}")
    
    return df

# Define features to use in the model
def get_feature_columns(df):
    """Get the list of feature columns to use in the model."""
    # Basic features
    basic_features = ['DayOfYear', 'Month', 'DayOfWeek', 'Price', 'PromotionFlag', 'Seasonality', 'External']
    
    # Add lag features if they exist
    all_features = basic_features.copy()
    for col in df.columns:
        if 'lag' in col or 'rolling' in col:
            all_features.append(col)
    
    return all_features

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
        features = get_feature_columns(df)
        X = df[features]
        y = df['Sales Quantity']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        joblib.dump(scaler, SCALER_PATH)
        
        # Train model with optimized parameters (based on evaluation results)
        model = XGBRegressor(
            n_estimators=200,           # More trees for better performance
            learning_rate=0.05,         # Lower learning rate for better generalization
            max_depth=6,                # Control model complexity
            min_child_weight=2,         # Helps prevent overfitting
            subsample=0.8,              # Use 80% of data for each tree
            colsample_bytree=0.8,       # Use 80% of features for each tree
            gamma=0.1,                  # Minimum loss reduction for split
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            n_jobs=-1,                  # Use all available cores
            random_state=42             # For reproducibility
        )
        
        # Fit the model - compatible with all XGBoost versions
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log evaluation metrics
        logger.info(f"Model Evaluation Metrics:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        
        # Save evaluation metrics
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        # Add feature importance if available
        try:
            feature_importance = model.feature_importances_
            metrics['feature_importance'] = dict(zip(features, feature_importance.tolist()))
        except:
            logger.warning("Could not extract feature importance")
        
        # Save metrics to file
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        
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
        
        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            # If no scaler exists, create a new one
            scaler = StandardScaler()
            scaler.fit(df[get_feature_columns(df)])
            joblib.dump(scaler, SCALER_PATH)
        
        # Define features
        features = get_feature_columns(df)
        
        # Create next day prediction data
        next_day = pd.DataFrame([
            {
                'DayOfYear': df['DayOfYear'].max() + 1,
                'Month': (pd.Timestamp.now() + pd.Timedelta(days=1)).month,
                'DayOfWeek': (pd.Timestamp.now() + pd.Timedelta(days=1)).dayofweek,
                'Price': df['Price'].mean(),
                'PromotionFlag': 1,  # Assuming promotion is active
                'Seasonality': 2,    # Using a default seasonality value
                'External': 1         # Using a default external factor value
            }
        ])
        
        # Add lag features if they exist in the model
        for col in features:
            if col not in next_day.columns and ('lag' in col or 'rolling' in col):
                if 'lag1' in col:
                    # Use the most recent value
                    source_col = col.split('_lag')[0]
                    next_day[col] = df[source_col].iloc[-1]
                elif 'lag7' in col:
                    # Use the value from 7 days ago
                    source_col = col.split('_lag')[0]
                    next_day[col] = df[source_col].iloc[-7] if len(df) > 7 else df[source_col].mean()
                elif 'rolling_mean' in col:
                    # Use the average of the last 7 days
                    source_col = col.split('_rolling')[0]
                    next_day[col] = df[source_col].tail(7).mean()
        
        # Scale the features
        next_day_scaled = scaler.transform(next_day[features])
        
        # Make prediction
        prediction = model.predict(next_day_scaled)[0]
        result = round(float(prediction), 2)  # Ensure it's a native Python float
        
        logger.info(f"Demand forecast: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in demand forecasting: {e}")
        # Return a reasonable default in case of error
        return 0.0
