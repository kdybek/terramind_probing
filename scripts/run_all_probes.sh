jobid=$(sbatch --parsable scripts/probe.sh)
sbatch --dependency=afterok:$jobid collate_res.sh
