from agents.llm_agent import ask_llm
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        
        # Format the data for better LLM processing
        formatted_inventory = json.dumps(inventory_alerts[:5], indent=2) if inventory_alerts else "No alerts"
        formatted_pricing = json.dumps(pricing_suggestions[:5], indent=2) if pricing_suggestions else "No suggestions"
        
        # Create a more structured prompt for better results
        prompt = f"""
        As an AI decision-making agent, you are responsible for coordinating inventory optimization.
        
        ## Current Data:
        
        ### Forecasted Demand: 
        {demand}
        
        ### Inventory Alerts (showing up to 5):
        {formatted_inventory}
        {'...' if len(inventory_alerts) > 5 else ''}
        
        ### Pricing Suggestions (showing up to 5):
        {formatted_pricing}
        {'...' if len(pricing_suggestions) > 5 else ''}
        
        ## Instructions:
        
        Please provide a comprehensive plan with the following sections:
        
        1. Products to Reorder: Identify which products need immediate reordering based on inventory alerts
        2. Price Adjustments: Recommend which product prices should be adjusted based on pricing suggestions
        3. Strategic Recommendations: Provide 2-3 strategic recommendations for inventory optimization
        
        Format your response in a clear, structured manner with headings for each section.
        """
        
        # Get response from LLM
        response = ask_llm(prompt)
        
        logger.info("Decision plan generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in decision making: {e}")
        return "Error: Unable to generate decision plan. Please check the logs for details."
