jobid=$(sbatch --parsable scripts/probe.sh)
sbatch --dependency=afterok:$jobid scripts/collate_res.sh
