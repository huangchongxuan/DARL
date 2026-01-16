#!/bin/bash

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# echo "local rank is: $RANK"
# if [ "$RANK" == "0" ]; then
#     ray start --head --port=$MASTER_PORT --num-gpus=8
# else
#     ray start --address=$MASTER_ADDR:$MASTER_PORT --num-gpus=8
# fi

CPUS_PER_TASK=80

if [ "$RANK" == "0" ]; then
    echo "Starting Ray head node on $MASTER_ADDR"
    ray start --head \
        --port=$MASTER_PORT \
        --num-cpus="${CPUS_PER_TASK}" \
        --num-gpus=8 \
        --verbose \
        --block &
        
    # Get the Ray address
    ray_address="ray://${MASTER_ADDR}:${MASTER_PORT}"

    # Wait for Ray to start
    echo running script: $1
    # sleep 120
    bash $1
else
    sleep 120
    node_ip_address=`ip route get 1.2.3.4 | awk '{print $7}'`
    echo "ip is $node_ip_address"
    # Start Ray worker nodes
    echo "Starting Ray worker node on $node_ip_address"
    ray start --address="$MASTER_ADDR:$MASTER_PORT" \
        --num-cpus="${CPUS_PER_TASK}" \
        --verbose \
        --block
fi 

echo 'end worker.start.sh, should NEVER happen on ANY worker!'