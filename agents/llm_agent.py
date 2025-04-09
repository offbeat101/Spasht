import subprocess
import logging
import time
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_agent')

# Cache LLM responses to avoid redundant calls for identical prompts
@lru_cache(maxsize=100)
def cached_llm_call(prompt_hash, model="tinyllama"):
    """Cached version of LLM call to avoid redundant processing.
    
    Args:
        prompt_hash (str): A hash of the prompt to use as cache key
        model (str): The model to use for generation
        
    Returns:
        str: The LLM response
    """
    # This is just for the lru_cache - actual prompt is passed to ask_llm
    return None

def ask_llm(prompt, model="tinyllama", timeout=30):
    """Ask the LLM a question and get a response.
    
    Args:
        prompt (str): The prompt to send to the LLM
        model (str, optional): The model to use. Defaults to "tinyllama".
        timeout (int, optional): Timeout in seconds. Defaults to 30.
        
    Returns:
        str: The LLM response
    """
    try:
        # Create a hash of the prompt for caching
        prompt_hash = hash(prompt)
        
        # Check if we have a cached response
        cached_response = cached_llm_call.__wrapped__(prompt_hash, model)
        if cached_response is not None:
            logger.info(f"Using cached LLM response for prompt hash {prompt_hash}")
            return cached_response
        
        logger.info(f"Sending prompt to {model}")
        start_time = time.time()
        
        # Run the LLM with timeout
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        
        # Process the response
        response = result.stdout.decode("utf-8")
        
        # Log performance metrics
        elapsed_time = time.time() - start_time
        logger.info(f"LLM response received in {elapsed_time:.2f} seconds")
        
        # Cache the response
        cached_llm_call.__wrapped__(prompt_hash, model)
        
        return response
        
    except subprocess.TimeoutExpired:
        logger.error(f"LLM request timed out after {timeout} seconds")
        return "Error: LLM request timed out. Please try again."
        
    except Exception as e:
        logger.error(f"Error in LLM request: {e}")
        return f"Error: Unable to get LLM response. {str(e)}"
