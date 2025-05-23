# --- Part 2: Basic Logging with Python's logging module ---
import logging
import os

# Configure logging
# Create a logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "ml_course_app.log")

# Basic configuration:
# - Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
# - Format: How the log messages will look
# - Handlers: Where the log messages go (e.g., file, console)

logging.basicConfig(
    level=logging.INFO, # Log messages of INFO level and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE), # Log to a file
        logging.StreamHandler()        # Log to the console
    ]
)

# Get a logger instance (best practice to use __name__ for module-level logger)
logger = logging.getLogger(__name__)

# If you want a specific logger for a component:
# data_loader_logger = logging.getLogger("data_loader")
# model_trainer_logger = logging.getLogger("model.trainer")

print("\n--- Python Logging ---")
logger.debug("This is a debug message. (Will not be shown with INFO level)") # Not shown due to level=INFO
logger.info("Application started. This is an informational message.")
logger.warning("A potential issue was detected, but the application can continue.")
logger.error("An error occurred. Something went wrong.")
logger.critical("A critical error occurred. The application might be unable to continue.")

# Example usage in a function
def process_data_with_logging(data_item):
    logger.info(f"Starting to process data item: {data_item}")
    try:
        # Simulate some processing
        if not isinstance(data_item, dict):
            logger.error(f"Invalid data type for item: {type(data_item)}. Expected dict.")
            raise TypeError("Data item must be a dictionary.")
        result = data_item.get("value", 0) * 2
        logger.info(f"Successfully processed data item. Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing data item {data_item}: {e}", exc_info=True) # exc_info=True logs stack trace
        # raise # Optionally re-raise the exception

process_data_with_logging({"value": 10})
process_data_with_logging(None) # This will cause an error

print(f"\nLog messages are being written to console and to '{LOG_FILE}'.")
print("Check the log file for more details.")

