#!/bin/bash

# Get a list of all queues
queues=$(rabbitmqctl list_queues | awk '{print $1}' | tail -n +2)

# Loop through each queue and delete it
for queue in $queues; do
    echo "Deleting queue: $queue"
    rabbitmqctl delete_queue "$queue"
done

brew services restart rabbitmq
