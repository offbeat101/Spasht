import random
import numpy as np
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        return "Error" if batch_size == 1 else {"Error": batch_size}
