#!/bin/bash

# Detect `python3` command.
# This workaround addresses a common issue:
#   `python` points to `python2`, which is deprecated.
export PYTHONS
export RVAL

PYTHONS=$(compgen -c | grep "^python3$")

# `$?` is a built-in variable in bash, which is the exit status of the most
# recently-executed command; by convention, 0 means success and anything else
# indicates failure.
RVAL=$?

if [[ $RVAL -eq 0 ]]; then  # if `python3` exist
    PYTHON="python3"
else
    PYTHON="python"
fi

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script helps launch distributed training job on local machine."
    echo
    echo "Usage: $0 GPUS COMMAND [ARGS]"
    echo
    echo "Example: $0 8 ddpm [--help]"
    echo
    echo "To enable multi-node training, one can reuse this script" \
         "by simply setting the following environment variables on each" \
         "machine:"
    echo "    MASTER_IP: The IP address of the master node."
    echo "    MASTER_PORT: The port of the master node."
    echo "    NODE_SIZE: Number of nodes (machines) used for training."
    echo "    NODE_RANK: Node rank of the current machine."
    echo
    echo "NOTE: In multi-node training, \`GPUS\` refers to the number" \
         "of GPUs on each local machine, or say, GPUs per node."
    echo
    echo "Example of using 16 GPUs on 2 machines (i.e., 8 GPUs each):"
    echo
    echo "    On machine 0: MASTER_IP=node_0_ip MASTER_PORT=node_0_port" \
         "NODE_SIZE=2 NODE_RANK=0 $0 8 ddpm [--help]"
    echo "    On machine 1: MASTER_IP=node_0_ip MASTER_PORT=node_0_port" \
         "NODE_SIZE=2 NODE_RANK=1 $0 8 ddpm [--help]"
    echo
    echo "Detailed instruction on available commands:"
    echo "--------------------------------------------------"
    ${PYTHON} ./main.py --help
    echo
    exit 0
fi

GPUS=$1
COMMAND=$2

# Help message for a particular command.
if [[ $# -lt 3 || ${*: -1} == "--help" ]]; then
    echo "Detailed instruction on the arguments for command \`"${COMMAND}"\`:"
    echo "--------------------------------------------------"
    ${PYTHON} ./main.py ${COMMAND} --help
    echo
    exit 0
fi

# Switch memory allocator if available.
# Search order: jemalloc.so -> tcmalloc.so.
# According to https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html,
# it can get better performance by reusing memory as much as possible than
# default malloc function.
JEMALLOC=$(ldconfig -p | grep -i "libjemalloc.so$" | tr " " "\n" | grep "/" \
           | head -n 1)
TCMALLOC=$(ldconfig -p | grep -i "libtcmalloc.so.4$" | tr " " "\n" | grep "/" \
           | head -n 1)
if [ -n "$JEMALLOC" ]; then  # if found the path to libjemalloc.so
    echo "Switch memory allocator to jemalloc."
    export LD_PRELOAD=$JEMALLOC:$LD_PRELOAD
elif [ -n "$TCMALLOC" ]; then  # if found the path to libtcmalloc.so.4
    echo "Switch memory allocator to tcmalloc."
    export LD_PRELOAD=$TCMALLOC:$LD_PRELOAD
fi

# Get an available port for launching distributed training.
# Credit to https://superuser.com/a/1293762.
export DEFAULT_FREE_PORT
DEFAULT_FREE_PORT=$(comm -23 <(seq 49152 65535 | sort) \
                    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
                    | shuf | head -n 1)

MASTER_IP=${MASTER_IP:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$DEFAULT_FREE_PORT}
NODE_SIZE=${NODE_SIZE:-1}
NODE_RANK=${NODE_RANK:-0}

${PYTHON} -m torch.distributed.launch \
    --master_addr=${MASTER_IP} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NODE_SIZE} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${GPUS} \
    ./main.py \
        ${COMMAND} \
        ${@:3} \
        || exit 1  # Stop the script when it finds exception threw by Python.
