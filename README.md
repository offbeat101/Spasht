# RetailSync: Multi-Agent AI System for Intelligent Inventory Optimization

## Overview

RetailSync is a multi-agent AI system designed to optimize retail inventory management through coordinated decision-making. The system predicts demand, monitors inventory levels, optimizes pricing, and generates strategic recommendations to help retail businesses maintain optimal inventory levels, avoid stockouts, and maximize profits.

## Problem Statement

Retail businesses face significant challenges in inventory management:

- **Inefficient Demand Forecasting**: Reliance on manual calculations and basic models
- **Suboptimal Inventory Levels**: Stockouts and overstocking issues
- **Reactive Pricing Strategies**: Manual price adjustments that don't respond quickly enough
- **Fragmented Communication**: Siloed information between stores, warehouses, and suppliers
- **Limited Data Utilization**: Underutilization of available data for strategic decisions

## Solution Architecture

RetailSync addresses these challenges through a coordinated multi-agent AI system:

### Multi-Agent Architecture

- **Demand Forecasting Agent**: Predicts future product demand using XGBoost
- **Inventory Monitoring Agent**: Tracks stock levels and generates prioritized reorder alerts
- **Pricing Optimization Agent**: Dynamically adjusts product pricing based on elasticity
- **Decision Coordination Agent**: Integrates insights from all agents to create action plans
- **Customer Simulation Agent**: Models customer purchasing behavior in response to price changes

### Key Features

- Advanced machine learning for accurate demand forecasting
- Vectorized operations for efficient data processing
- Intelligent reorder recommendations with urgency classification
- Dynamic pricing optimization based on elasticity and competition
- Customer purchase simulation for testing pricing strategies
- Comprehensive visualization dashboard

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/retailsync.git
cd retailsync
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Initialize the database with sample data:
```
python utils/init_db.py
```

4. Run the application:
```
streamlit run app.py
```

## Usage

1. Configure settings in the sidebar:
   - Adjust daily sales rate for inventory monitoring
   - Select customer segment for simulation
   - Choose whether to retrain the demand forecasting model

2. Click "Run Analysis" to generate:
   - Demand forecasts
   - Inventory alerts
   - Pricing recommendations
   - Strategic decision plan

3. Explore the detailed tabs:
   - View inventory alerts with urgency levels
   - Review price recommendations with expected impact
   - Read the comprehensive decision plan
   - Simulate customer responses to price changes

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SHARED DATA LAYER                       │
│  (Inventory Records, Sales History, Supplier Information)   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       AGENT LAYER                           │
│                                                             │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────┐│
│  │    Demand     │◄────►│   Inventory   │◄────►│  Pricing  ││
│  │  Forecasting  │      │   Monitoring  │      │   Agent   ││
│  │     Agent     │      │     Agent     │      │           ││
│  └───────┬───────┘      └───────┬───────┘      └─────┬─────┘│
│          │                      │                    │      │
│          │                      ▼                    │      │
│          │        ┌───────────────────────┐          │      │
│          └───────►│      Decision         │◄─────────┘      │
│                   │       Agent           │                 │
│                   └───────────┬───────────┘                 │
│                               │                             │
│                               ▼                             │
│                   ┌───────────────────────┐                 │
│                   │      Customer         │                 │
│                   │     Simulation        │                 │
│                   │       Agent           │                 │
│                   └───────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                     │
│      (Dashboards, Alerts, Recommendations, Controls)        │
└─────────────────────────────────────────────────────────────┘
```

## Testing

Run the test suite to verify agent functionality:

```
python -m unittest discover tests
```

## Performance Optimizations

The system has been optimized for speed and efficiency:

- Added caching with `@lru_cache` to avoid redundant database queries
- Used vectorized operations instead of slow loops
- Implemented batch processing for customer simulations
- Added parallel processing for model training
- Optimized database queries to only fetch required columns

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- XGBoost team for their gradient boosting framework
- Streamlit team for their interactive web application framework
- Pandas and NumPy teams for their data processing libraries
