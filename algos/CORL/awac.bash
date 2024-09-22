#!/bin/bash

# Array of reward types
reward_types=("linear" "mixed" "real" "virtual")

# Array of seeds
seeds=(0 1 2 3 4)

# Array of GPU IDs
gpu_ids=("0" "1" "2" "3")

# Function to run awac_virtual
run_awac_virtual() {
    reward_type=$1
    gpu_id=$2
    seed=$3
    CUDA_VISIBLE_DEVICES="$gpu_id" python algos/CORL/awac_virtual.py reward_type=$reward_type seed=$seed
}


# Loop through each reward type and run awac_virtual with different GPUs and seeds in parallel
for i in "${!seeds[@]}"
do
    seed="${seeds[$i]}"
    for j in "${!reward_types[@]}"
    do
        reward_type="${reward_types[$j]}"
        gpu_id="${gpu_ids[$j]}"
        run_awac_virtual "$reward_type" "$gpu_id" "$seed" &
    done
    wait
done

# Wait for all background processes to finish
wait