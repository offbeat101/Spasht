import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import os
import json
from datetime import datetime

# Import agents
from agents.demand_agent import forecast_demand, train_demand_model
from agents.inventory_agent import monitor_inventory
from agents.pricing_agent import optimize_pricing
from agents.decision_agent import make_decision
from agents.customer_agent import customer_response

# Configure logging
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retail_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('app')

# Set page configuration
st.set_page_config(
    page_title="Retail Inventory Optimization",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {text-align: center; margin-bottom: 30px;}
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    .alert-high {color: #ff4b4b;}
    .alert-medium {color: #ff9d45;}
    .price-decrease {color: #ff4b4b;}
    .price-increase {color: #0068c9;}
    .stButton button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Demand forecasting settings
    st.header("Demand Forecasting")
    retrain_model = st.checkbox("Retrain demand model", value=False)
    
    # Inventory monitoring settings
    st.header("Inventory Monitoring")
    daily_sales_rate = st.slider("Daily Sales Rate", min_value=1, max_value=50, value=10)
    
    # Customer simulation settings
    st.header("Customer Simulation")
    customer_segment = st.selectbox(
        "Customer Segment",
        options=[(1, "Price Sensitive"), (2, "Quality Focused"), (3, "Luxury")],
        format_func=lambda x: x[1]
    )[0]
    
    # Run button
    st.header("Actions")
    run_button = st.button("🚀 Run Analysis", key="run_analysis")
    
    # About section
    st.header("About")
    st.info(
        "This multi-agent system optimizes retail inventory management by coordinating between demand forecasting, "
        "inventory monitoring, and pricing optimization agents."
    )

# Main content
st.markdown("<h1 class='main-header'>🛒 Retail Inventory Optimization - Multi-Agent System</h1>", unsafe_allow_html=True)

# Initialize session state for storing results
if 'demand' not in st.session_state:
    st.session_state.demand = None
if 'inventory_alerts' not in st.session_state:
    st.session_state.inventory_alerts = None
if 'price_updates' not in st.session_state:
    st.session_state.price_updates = None
if 'decision' not in st.session_state:
    st.session_state.decision = None
if 'last_run' not in st.session_state:
    st.session_state.last_run = None

# Function to run all agents
def run_analysis():
    start_time = time.time()
    
    with st.spinner("🔮 Forecasting demand..."):
        # Train model if requested
        if retrain_model:
            with st.status("Training demand forecasting model...", expanded=True) as status:
                train_demand_model(force_retrain=True)
                status.update(label="Model training complete!", state="complete")
        
        # Forecast demand
        st.session_state.demand = forecast_demand()
    
    with st.spinner("📦 Monitoring inventory..."):
        # Monitor inventory with configured daily sales rate
        st.session_state.inventory_alerts = monitor_inventory(daily_sales_rate=daily_sales_rate)
    
    with st.spinner("💰 Optimizing pricing..."):
        # Optimize pricing
        st.session_state.price_updates = optimize_pricing()
    
    with st.spinner("🤖 Generating decision plan..."):
        # Generate decision plan
        st.session_state.decision = make_decision(
            st.session_state.demand,
            st.session_state.inventory_alerts,
            st.session_state.price_updates
        )
    
    # Record timestamp
    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log performance
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

# Run analysis when button is clicked
if run_button:
    run_analysis()

# Display results if available
if st.session_state.last_run:
    st.success(f"Last analysis run: {st.session_state.last_run}")
    
    # Create 3-column layout for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("📈 Demand Forecast")
        if st.session_state.demand is not None:
            st.metric("Predicted Demand", f"{st.session_state.demand:,.2f} units")
            
            # Display model metrics if available
            try:
                with open('models/demand_model_metrics.json', 'r') as f:
                    metrics = json.load(f)
                    
                with st.expander("Model Evaluation Metrics"):
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                        st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
                    with metric_col2:
                        st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
                        st.metric("Mean Squared Error", f"{metrics['mse']:.2f}")
                    
                    # Display feature importance
                    if 'feature_importance' in metrics:
                        st.subheader("Feature Importance")
                        # Convert to DataFrame for better display
                        fi_df = pd.DataFrame({
                            'Feature': list(metrics['feature_importance'].keys()),
                            'Importance': list(metrics['feature_importance'].values())
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig = {
                            "data": [{
                                "x": fi_df['Importance'][:10],
                                "y": fi_df['Feature'][:10],
                                "type": "bar",
                                "orientation": "h",
                                "marker": {"color": "#0068c9"}
                            }],
                            "layout": {
                                "title": "Top 10 Features by Importance",
                                "xaxis": {"title": "Importance"},
                                "yaxis": {"title": "Feature"}
                            }
                        }
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Model metrics not available. Retrain the model to see metrics.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("📦 Inventory Status")
        if st.session_state.inventory_alerts is not None:
            alert_count = len(st.session_state.inventory_alerts)
            st.metric("Products Needing Reorder", alert_count)
            
            # Count urgency levels
            if alert_count > 0:
                high_urgency = sum(1 for alert in st.session_state.inventory_alerts if alert.get('Urgency') == 'High')
                st.markdown(f"<span class='alert-high'>High Urgency: {high_urgency}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("💰 Pricing Updates")
        if st.session_state.price_updates is not None:
            price_changes = sum(1 for item in st.session_state.price_updates if item['Price'] != item['Suggested Price'])
            st.metric("Products with Price Changes", price_changes)
            
            # Calculate average price change
            if price_changes > 0:
                avg_change = np.mean([item.get('Price Change %', 0) for item in st.session_state.price_updates 
                                     if item['Price'] != item['Suggested Price']])
                change_text = f"Average Change: {avg_change:.2f}%"
                if avg_change < 0:
                    st.markdown(f"<span class='price-decrease'>{change_text}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='price-increase'>{change_text}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed sections with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Inventory Alerts", "💰 Price Recommendations", "🤖 Decision Plan", "🧪 Customer Simulation"])
    
    with tab1:
        if st.session_state.inventory_alerts:
            # Convert to DataFrame for better display
            df_inventory = pd.DataFrame(st.session_state.inventory_alerts)
            
            # Add styling
            def highlight_urgency(val):
                if val == 'High':
                    return 'background-color: #ffcccc'
                elif val == 'Medium':
                    return 'background-color: #ffffcc'
                return ''
            
            # Display styled dataframe
            st.dataframe(
                df_inventory.style.applymap(highlight_urgency, subset=['Urgency']),
                use_container_width=True
            )
            
            # Download button
            csv = df_inventory.to_csv(index=False)
            st.download_button(
                label="Download Inventory Alerts",
                data=csv,
                file_name=f"inventory_alerts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No inventory alerts to display. Run the analysis first.")
    
    with tab2:
        if st.session_state.price_updates:
            # Convert to DataFrame for better display
            df_pricing = pd.DataFrame(st.session_state.price_updates)
            
            # Add styling
            def highlight_price_change(val):
                if val < 0:
                    return 'color: red'
                elif val > 0:
                    return 'color: green'
                return ''
            
            # Display styled dataframe
            st.dataframe(
                df_pricing.style.applymap(highlight_price_change, subset=['Price Change %']),
                use_container_width=True
            )
            
            # Download button
            csv = df_pricing.to_csv(index=False)
            st.download_button(
                label="Download Price Recommendations",
                data=csv,
                file_name=f"price_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No price updates to display. Run the analysis first.")
    
    with tab3:
        if st.session_state.decision:
            st.markdown("### Decision Plan")
            st.markdown(st.session_state.decision)
            
            # Download button
            st.download_button(
                label="Download Decision Plan",
                data=st.session_state.decision,
                file_name=f"decision_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No decision plan to display. Run the analysis first.")
    
    with tab4:
        st.markdown("### Customer Purchase Simulation")
        
        # Customer simulation controls
        sim_col1, sim_col2 = st.columns(2)
        
        with sim_col1:
            sim_price = st.number_input("Product Price ($)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
        
        with sim_col2:
            sim_stock = st.number_input("Available Stock", min_value=0, max_value=1000, value=100, step=10)
        
        sim_batch_size = st.slider("Number of Customers", min_value=1, max_value=1000, value=100)
        
        # Run simulation button
        if st.button("Run Customer Simulation"):
            with st.spinner("Simulating customer behavior..."):
                # Run customer simulation
                result = customer_response(sim_price, sim_stock, customer_segment, sim_batch_size)
                
                # Display results
                if isinstance(result, dict):
                    # Create pie chart for batch results
                    labels = list(result.keys())
                    values = list(result.values())
                    
                    # Calculate purchase rate
                    purchase_rate = result.get("Buy", 0) / sim_batch_size * 100 if sim_batch_size > 0 else 0
                    
                    # Display metrics
                    st.metric("Purchase Rate", f"{purchase_rate:.1f}%")
                    
                    # Create columns for visualization
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Create pie chart
                        fig = {
                            "data": [{
                                "values": values,
                                "labels": labels,
                                "type": "pie",
                                "hole": 0.4,
                                "marker": {"colors": ["#0068c9", "#ff4b4b", "#ffab00"]}
                            }],
                            "layout": {"title": "Customer Decisions"}
                        }
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        # Create bar chart showing purchase rate by price
                        test_prices = np.linspace(max(1, sim_price - 50), sim_price + 50, 10)
                        purchase_rates = []
                        
                        for price in test_prices:
                            test_result = customer_response(price, sim_stock, customer_segment, 100)
                            if isinstance(test_result, dict):
                                rate = test_result.get("Buy", 0) / 100 * 100
                                purchase_rates.append(rate)
                            else:
                                purchase_rates.append(0)
                        
                        price_sensitivity = {
                            "data": [{
                                "x": test_prices,
                                "y": purchase_rates,
                                "type": "bar",
                                "marker": {"color": "#0068c9"}
                            }],
                            "layout": {
                                "title": "Price Sensitivity",
                                "xaxis": {"title": "Price ($)"},
                                "yaxis": {"title": "Purchase Rate (%)"},
                            }
                        }
                        st.plotly_chart(price_sensitivity, use_container_width=True)
                else:
                    # Display single customer result
                    st.info(f"Customer decision: {result}")

else:
    # Display instructions when first loading the app
    st.info("👈 Configure the settings in the sidebar and click 'Run Analysis' to start.")
    
    # Display sample images or explanations
    st.markdown("""
    ## How It Works
    
    This multi-agent system uses several specialized AI agents working together:
    
    1. **Demand Forecasting Agent** - Predicts future product demand using machine learning
    2. **Inventory Monitoring Agent** - Tracks inventory levels and generates alerts for products that need reordering
    3. **Pricing Optimization Agent** - Suggests optimal pricing based on elasticity and competitor prices
    4. **Decision Agent** - Coordinates between other agents to make strategic inventory decisions
    5. **Customer Agent** - Simulates customer purchase behavior based on price and stock availability
    
    The system helps retail businesses maintain optimal inventory levels, avoid stockouts, and maximize profits.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Retail Inventory Optimization System | Developed with Streamlit and Python"
    "</div>",
    unsafe_allow_html=True
)
