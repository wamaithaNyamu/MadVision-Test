#!/bin/bash

echo "Starting the cleanup and application services..."

# Delete RabbitMQ queues (or any other 'delete_queues.sh' functionality)
echo "Deleting RabbitMQ queues..."
(./delete_queues.sh >> delete_queues.log 2>&1 &)
echo "Queues deletion initiated. Check delete_queues.log for details."

# Start FastAPI in development mode
echo "Starting FastAPI app in development mode..."
(uvicorn app:app --reload >> fastapi.log 2>&1 &)
echo "FastAPI app initiated. Check fastapi.log for details."

# Run the consumer script
echo "Running consumer script..."
(./run_consumer.sh >> consumer.log 2>&1 &)
echo "Consumer script initiated. Check consumer.log for details."

# Run Celery worker
echo "Starting Celery worker..."
(./run_celery.sh >> celery.log 2>&1 &)
echo "Celery worker initiated. Check celery.log for details."

echo "All processes started successfully in the background!"
