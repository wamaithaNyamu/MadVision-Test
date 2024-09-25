import logging

# Create a logger
logger = logging.getLogger(__name__)

# Configure the logger
logger.setLevel(logging.INFO)  # Set the default logging level

# Create a console handler
console_handler = logging.StreamHandler()

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Avoid propagating messages to the root logger
logger.propagate = False
