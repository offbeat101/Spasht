# Retail Inventory Optimization - Multi-Agent System (Colab Version)

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

## Colab-Specific Features

This version includes several features specific to running in Google Colab:

1. **ngrok Integration**: Uses ngrok to expose the Streamlit app to the internet
2. **Automatic Setup**: Includes a setup script to create all necessary files and directories
3. **Sample Data Generation**: Generates sample data for testing
4. **Simplified Decision Agent**: Uses a simulated decision agent instead of an LLM to avoid dependencies

## Folder Structure

- `retail_inventory_optimization.ipynb`: Main Colab notebook
- `colab_app.py`: Streamlit application
- `colab_setup.py`: Setup script to create files and directories
- `agents/`: Directory containing all agent implementations
- `utils/`: Directory containing utility scripts
- `db/`: Directory for the SQLite database
- `models/`: Directory for saved ML models
- `logs/`: Directory for log files
- `data/`: Directory for CSV data files
