#!/bin/bash
set -e

# Sync local gcsfs/ over to the VM's ~/gcsfs/ directory
tar -czf - gcsfs/ | /usr/bin/ssh -i /usr/local/google/home/yuxinj/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Hostname=nic0.test-node-simulator.us-central1-b.c.gcs-aiml-clients-testing-101.internal.gcpnode.com yuxinj@yuxinj-test-benchmark 'cd ~/gcsfs && tar -xzf -'

# Run the benchmark (128 processes, 1 thread, 1 round)
OUTPUT1=$(/usr/bin/ssh -i /usr/local/google/home/yuxinj/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Hostname=nic0.test-node-simulator.us-central1-b.c.gcs-aiml-clients-testing-101.internal.gcpnode.com yuxinj@yuxinj-test-benchmark 'bash -l -c "source ~/miniconda/bin/activate && conda activate gcsfs_head && python3 ~/run_multiple.py --processes 128 --threads 1 --rounds 1"')

# Extract the Max Latency from the aggregate block
MAX_LATENCY1=$(echo "$OUTPUT1" | grep "Max Latency:" | awk '{print $3}')

# Run the benchmark (32 processes, 10 threads, 1 round)
OUTPUT2=$(/usr/bin/ssh -i /usr/local/google/home/yuxinj/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Hostname=nic0.test-node-simulator.us-central1-b.c.gcs-aiml-clients-testing-101.internal.gcpnode.com yuxinj@yuxinj-test-benchmark 'bash -l -c "source ~/miniconda/bin/activate && conda activate gcsfs_head && python3 ~/run_multiple.py --processes 32 --threads 10 --rounds 1"')

# Extract the Max Latency from the aggregate block
MAX_LATENCY2=$(echo "$OUTPUT2" | grep "Max Latency:" | awk '{print $3}')

# Output the maximum of the two latencies
python3 -c "print(max($MAX_LATENCY1, $MAX_LATENCY2))"
