import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agents
from agents.pricing_agent import optimize_pricing
from agents.inventory_agent import monitor_inventory
from agents.demand_agent import forecast_demand, train_demand_model
from agents.customer_agent import customer_response, calculate_purchase_probability
from agents.llm_agent import ask_llm
from agents.decision_agent import make_decision

class TestAgents(unittest.TestCase):
    
    @patch('agents.pricing_agent.get_pricing_data')
    def test_pricing_agent(self, mock_get_pricing_data):
        # Create mock data
        mock_df = pd.DataFrame({
            'Product ID': [1, 2, 3, 4],
            'Store ID': [10, 20, 30, 40],
            'Price': [50.0, 30.0, 20.0, 100.0],
            'Competitor Prices': [45.0, 35.0, 25.0, 90.0],
            'Elasticity Index': [1.2, 1.8, 2.0, 0.9]
        })
        
        # Configure the mock to return our test data
        mock_get_pricing_data.return_value = mock_df
        
        # Call the function
        result = optimize_pricing()
        
        # Assertions
        self.assertEqual(len(result), 4)
        
        # Products with elasticity > 1.5 should have suggested prices different from original
        self.assertEqual(result[0]['Suggested Price'], 50.0)  # Elasticity 1.2, no change
        self.assertNotEqual(result[1]['Suggested Price'], 30.0)  # Elasticity 1.8, should be reduced
        self.assertNotEqual(result[2]['Suggested Price'], 20.0)  # Elasticity 2.0, should be reduced
        self.assertEqual(result[3]['Suggested Price'], 100.0)  # Elasticity 0.9, no change
    
    @patch('agents.inventory_agent.get_inventory_data')
    def test_inventory_agent(self, mock_get_inventory_data):
        # Create mock data
        mock_df = pd.DataFrame({
            'Product ID': [1, 2, 3, 4],
            'Store ID': [10, 20, 30, 40],
            'Stock Levels': [100, 20, 5, 50],
            'Supplier Lead Time (days)': [5, 10, 3, 7],
            'Reorder Point': [30, 30, 10, 20],
            'Safety Stock': [20, 15, 5, 10]
        })
        
        # Configure the mock to return our test data
        mock_get_inventory_data.return_value = mock_df
        
        # Call the function with default daily sales rate of 10
        result = monitor_inventory()
        
        # Assertions
        # Product 2 and 3 should trigger alerts (stock < lead time * daily rate)
        self.assertEqual(len(result), 2)
        
        # Check that the alerts contain the right products
        product_ids = [alert['Product ID'] for alert in result]
        self.assertIn(2, product_ids)
        self.assertIn(3, product_ids)
        
        # Check that urgency is calculated correctly
        for alert in result:
            if alert['Product ID'] == 3:  # Very low stock
                self.assertEqual(alert['Urgency'], 'High')
    
    def test_customer_agent(self):
        # Test out of stock
        result = customer_response(price=20, stock=0)
        self.assertEqual(result, "No purchase")
        
        # Test batch processing
        batch_result = customer_response(price=20, stock=100, batch_size=100)
        self.assertIsInstance(batch_result, dict)
        self.assertIn("Buy", batch_result)
        self.assertIn("No purchase", batch_result)
        self.assertEqual(batch_result["Buy"] + batch_result["No purchase"], 100)
        
        # Test probability calculation
        prob = calculate_purchase_probability(20, 1)  # Low price for price-sensitive segment
        self.assertGreater(prob, 0.5)  # Should be high probability
        
        prob = calculate_purchase_probability(120, 1)  # High price for price-sensitive segment
        self.assertLess(prob, 0.5)  # Should be low probability
    
    @patch('agents.llm_agent.subprocess.run')
    def test_llm_agent(self, mock_subprocess_run):
        # Configure the mock
        mock_process = MagicMock()
        mock_process.stdout = b"This is a test response"
        mock_subprocess_run.return_value = mock_process
        
        # Call the function
        result = ask_llm("Test prompt")
        
        # Assertions
        self.assertEqual(result, "This is a test response")
        mock_subprocess_run.assert_called_once()
    
    @patch('agents.decision_agent.ask_llm')
    def test_decision_agent(self, mock_ask_llm):
        # Configure the mock
        mock_ask_llm.return_value = "Test decision"
        
        # Call the function
        result = make_decision(100, [{"Product ID": 1}], [{"Product ID": 2}])
        
        # Assertions
        self.assertEqual(result, "Test decision")
        mock_ask_llm.assert_called_once()
    
    @patch('agents.demand_agent.get_demand_data')
    @patch('agents.demand_agent.joblib.load')
    @patch('agents.demand_agent.os.path.exists')
    def test_demand_agent(self, mock_exists, mock_joblib_load, mock_get_demand_data):
        # Configure the mocks
        mock_exists.return_value = True
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([150.5])
        mock_joblib_load.return_value = mock_model
        
        mock_df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Price': [10.0] * 10,
            'Promotions': ['Yes'] * 5 + ['No'] * 5,
            'Seasonality Factors': ['High'] * 10,
            'External Factors': ['Normal'] * 10,
            'Sales Quantity': [100.0] * 10,
            'Product ID': [1] * 10,
            'Store ID': [1] * 10
        })
        mock_get_demand_data.return_value = mock_df
        
        # Call the function
        result = forecast_demand()
        
        # Assertions
        self.assertEqual(result, 150.5)
        mock_model.predict.assert_called_once()

if __name__ == '__main__':
    unittest.main()
