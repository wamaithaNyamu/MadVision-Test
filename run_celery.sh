#!/bin/bash

# Directory containing your Celery files
CELERY_DIR="Celery"

# Start a Celery worker for each Python file in the Celery directory
for celery_file in "$CELERY_DIR"/celery_*.py; do
    if [[ -f "$celery_file" ]]; then
        # Extract the module name (without .py extension)
        module_name=$(basename "$celery_file" .py)

        echo "Starting Celery worker for $module_name"
        # Notice how we now pass only the module name relative to the root folder
        celery -A "$CELERY_DIR"."$module_name" worker --loglevel=info &  # Run in background
    fi
done

# Wait for all background processes to finish
wait

echo "All Celery workers have been started."
