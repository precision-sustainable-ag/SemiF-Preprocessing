#!/bin/bash

# Read batch IDs from Hydra config using Python
batch_ids=($(python src/automate_preprocess.py))

# Log file setup
LOG_FILE="scripts/batch_processing.log"
touch "$LOG_FILE"

# Processing loop
for batch_id in "${batch_ids[@]}"; do
    # Run the Python command
    echo "[$(date +"%Y-%m-%d %T")] Processing batch: $batch_id" | tee -a "$LOG_FILE"

    if python main.py batch_id="$batch_id"; then
        # Log success
        echo "[$(date +"%Y-%m-%d %T")] SUCCESS: Batch $batch_id processed" | tee -a "$LOG_FILE"
    else
        # Log error and continue
        echo "[$(date +"%Y-%m-%d %T")] ERROR: Failed to process batch $batch_id" | tee -a "$LOG_FILE"
    fi

    # Add spacing between batches in log
    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
done

echo "Processing complete. Results logged to $LOG_FILE"
