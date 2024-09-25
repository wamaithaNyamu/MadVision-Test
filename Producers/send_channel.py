#!/usr/bin/env python
import json
import pika
from Utilities.utilities_logging import logger
from Utilities.utilities_data_conversion import dict_to_bytes
def send_to_queue(queue_name,message):
    try:
        logger.info(f"Adding {message} to {queue_name}")
        logger.info(f"Type of message is {type(message)}")
            # If the message is a dict, convert it to a JSON string first
        if isinstance(message, dict):
            message = dict_to_bytes(message)
            
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        logger.info("Connecting to rabbit mq")
        channel = connection.channel()
        logger.info("Connected ...")
        channel.queue_declare(queue=queue_name)
        logger.info("Queue declared...")
        channel.basic_publish(exchange='', routing_key=queue_name, body=message)
        
        logger.info(f"added {message} to {queue_name}")
        connection.close()
    except Exception as e:
        logger.error(f"Something went wrong while trying to add the message to the {queue_name}. The error is {e}")