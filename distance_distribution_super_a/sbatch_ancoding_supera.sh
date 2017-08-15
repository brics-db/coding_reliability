#!/bin/bash
#SBATCH -J gearshifftK80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=62000M # gpu2
#SBATCH --partition=gpu2
#SBATCH --exclusive
#SBATCH --array 1-16
#SBATCH -o slurmgpu2_array-%A_%a.out
#SBATCH -e slurmgpu2_array-%A_%a.err

k=$SLURM_ARRAY_TASK_ID

PROJ=$HOME/cuda-workspace/coding_reliability/distance_distribution_super_a
CODREL=$PROJ/release/codrel
RESULTS=$PROJ/results

if [ $k -eq 1 ]; then
    Ks=$(seq 4 19)
    LOW_A=3
    HIGH_A=65535
    M=0
fi

# 2<= k <= 8
if [ $k -gt 1 -a $k -lt 9]; then
    Ks=$((18+k)) # 20 .. 26
    LOW_A=3
    HIGH_A=$(( (1<<(18-2*k))-1 )) # 14, 12, .., 2
    M=0
fi

# 9<= k <= 15
if [ $k -gt 8 -a $k -lt 16]; then
    Ks=$((11+k)) # 20 .. 26
    LOW_A=$(( (1<<(32-2*k))-1 )) # 14, 12, .., 2
    HIGH_A=65536
    M=1001
fi

for K in $Ks; do
    if [ $M -eq 0 ]; then
        echo "$CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results"
        $CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results
    else
        echo "$CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results -m $M"
        $CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results -m $M
    fi
done
