import os,sys

from Utilities.utilities_logging import logger

def create_directory(directory_path):
    """
    Creates a directory if it doesn't exist.

    Args:
        directory_path (str): The path of the directory to create.
    """
    try:
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create the directory
            os.makedirs(directory_path)
            logger.info(f"Directory '{directory_path}' created successfully.")
            return directory_path
        else:
            logger.info(f"Directory '{directory_path}' already exists.")
            return directory_path
    except Exception as e:
        logger.error(f"An error occurred while creating the directory: {e}")

