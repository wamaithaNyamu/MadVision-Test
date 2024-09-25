#!/usr/bin/env python
import pika, sys, os
import threading
from Utilities.utilities_logging import logger
from Consumers.receive_channel import receive_channel
from Celery.celery_download_video import process_video
from Utilities.utilities_data_conversion import bytes_to_dict

queue_name = 'download_video'  # Replace with your actual queue name

def callback(ch, method, properties, body):
    try:
        logger.info("----------------- Now processing the video from download_video -----------------")
        logger.info(f"Received {body}")
        logger.info(f"The message is of type {type(body)}")
        logger.info(f"Now sending to celery for downloading ...")
        body = bytes_to_dict(body)
        logger.info(f"Body has been changed to type dict : {type(body)}")
        process_video(body)
        # Acknowledge the message after processing
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except pika.exceptions.StreamLostError as e:
        logger.error(f"RabbitMQ connection lost: {e}")
         # Reconnect to RabbitMQ
        channel = receive_channel(queue_name)
        if channel:
            logger.info("Reconnected to RabbitMQ")
        else:
            logger.error("Failed to reconnect to RabbitMQ. Exiting...")
            
    except Exception as e:  # Catch other potential errors
        logger.error(f"An error occurred: {e}")

def main():
    channel = receive_channel(queue_name)
    if channel is None:
        logger.error("Failed to create channel. Exiting...")
        return

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)

    logger.info(f' [*] Waiting for messages in {queue_name}. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
