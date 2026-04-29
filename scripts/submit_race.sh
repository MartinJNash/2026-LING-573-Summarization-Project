#!/bin/bash
# Submit inference jobs to multiple partitions and cancel the rest when one starts running.

JOB1=$(sbatch --parsable scripts/run_inference_hyak.sh)   # gpu-2080ti
JOB2=$(sbatch --parsable scripts/run_inference_ckpt.sh)   # ckpt-all

echo "Submitted: gpu-2080ti=$JOB1  ckpt-all=$JOB2"
echo "Watching for first job to start running..."

while true; do
    for JOB in $JOB1 $JOB2; do
        STATE=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
        if [[ "$STATE" == "RUNNING" ]]; then
            echo "Job $JOB is RUNNING — cancelling the others."
            for OTHER in $JOB1 $JOB2; do
                [[ "$OTHER" != "$JOB" ]] && scancel "$OTHER" && echo "Cancelled $OTHER"
            done
            echo "Done. Watch your winner: squeue -j $JOB"
            exit 0
        fi
    done
    sleep 30
done
