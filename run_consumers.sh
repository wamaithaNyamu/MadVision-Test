#!/bin/bash

# Directory containing the consumer files
CONSUMER_DIR="Consumers"

# Print the current working directory
echo "Current working directory: $(pwd)"

# Find and run all Python files starting with 'consumer_' in the specified directory
for consumer_file in "$CONSUMER_DIR"/consumer_*.py; do
    if [[ -f "$consumer_file" ]]; then
        echo "Starting consumer: $consumer_file"
        # Start the consumer in the background
        PYTHONPATH=. python "$consumer_file" &
    else
        echo "No consumer files found in $CONSUMER_DIR."
    fi
done

# Wait for all background processes to finish
wait

echo "All consumers have been started."
