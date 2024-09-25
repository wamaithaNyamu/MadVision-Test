#!/usr/bin/env python
import pika, sys, os
from Utilities.utilities_logging import logger


def receive_channel(queue_name,pretech_count=1):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost',heartbeat=600, blocked_connection_timeout=600))
        channel = connection.channel()
        # Declare the queue
        # channel.queue_declare(queue=queue_name, durable=True)
        channel.queue_declare(queue=queue_name)
        # Set prefetch to 3 so the consumer gets at most 3 unacknowledged messages at a time
        channel.basic_qos(prefetch_count=pretech_count)

        return channel
    except Exception as e:
        logger.error(f"Something went wrong while trying to set the channel  for {queue_name}")

