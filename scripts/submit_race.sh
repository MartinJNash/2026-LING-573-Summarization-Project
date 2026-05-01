#!/bin/bash
# Race inference jobs across partitions — cancel losers, chain eval on the winner.

INFERENCE_SCRIPTS=(
    "scripts/run_inference_hyak.sh"   # gpu-2080ti
    "scripts/run_inference_ckpt.sh"   # ckpt-all
)
EVAL_SCRIPT="scripts/run_eval_hyak.sh"

# Submit all inference jobs
declare -A JOB_LABELS
for SCRIPT in "${INFERENCE_SCRIPTS[@]}"; do
    PARTITION=$(grep -oP '(?<=--partition=)\S+' "$SCRIPT")
    JOB_ID=$(sbatch --parsable "$SCRIPT")
    JOB_LABELS[$JOB_ID]="$PARTITION"
    echo "Submitted $SCRIPT -> job $JOB_ID ($PARTITION)"
done

ALL_JOBS=("${!JOB_LABELS[@]}")
echo ""
echo "Racing: ${ALL_JOBS[*]}"
echo "Polling every 30s for first job to start running..."

WINNER=""
while [[ -z "$WINNER" ]]; do
    sleep 30
    for JOB in "${ALL_JOBS[@]}"; do
        STATE=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
        if [[ "$STATE" == "RUNNING" ]]; then
            WINNER=$JOB
            echo "Winner: job $WINNER (${JOB_LABELS[$WINNER]}) is RUNNING"
            break
        elif [[ -z "$STATE" ]]; then
            # Job ended before any ran (failed/cancelled) — remove from consideration
            echo "Job $JOB (${JOB_LABELS[$JOB]}) is gone — skipping."
            ALL_JOBS=("${ALL_JOBS[@]/$JOB}")
        fi
    done

    if [[ ${#ALL_JOBS[@]} -eq 0 ]]; then
        echo "All jobs failed or were cancelled before any started. Exiting."
        exit 1
    fi
done

# Cancel all other inference jobs
for JOB in "${ALL_JOBS[@]}"; do
    if [[ "$JOB" != "$WINNER" && -n "$JOB" ]]; then
        scancel "$JOB"
        echo "Cancelled job $JOB (${JOB_LABELS[$JOB]})"
    fi
done

# Submit eval as dependent on the winning inference job
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$WINNER "$EVAL_SCRIPT")
echo ""
echo "Submitted eval job $EVAL_JOB — will start after job $WINNER completes successfully."
echo ""
echo "Monitor inference: tail -f logs/${WINNER}.out"
echo "Monitor eval:      tail -f logs/${EVAL_JOB}.out"
echo "Watch queue:       watch -n 30 'squeue -u \$USER'"
