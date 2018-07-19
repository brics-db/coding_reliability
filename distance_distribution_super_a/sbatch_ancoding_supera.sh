#!/bin/bash
#SBATCH -J ANCoding-Small
#SBATCH -A p_ancoding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=1000M
#SBATCH --partition=gpu1,gpu2
# BATCH --array 1-21
#SBATCH --array 0
#SBATCH -o slurmgpu2_array-%A_%a.out
#SBATCH -e slurmgpu2_array-%A_%a.err

i=$SLURM_ARRAY_TASK_ID

PROJ=$HOME/cuda-workspace/coding_reliability/distance_distribution_super_a
CODREL=$PROJ/release/codrel
RESULTS=$PROJ/results

## rerun k = 19
if [ $i -eq 0 ]; then
    Ks=19
    LOW_A=3
    HIGH_A=65535
    M=0
fi

##
## k = 4..19
##
if [ $i -eq 1 ]; then
    Ks=$(seq 4 19)
    LOW_A=3
    HIGH_A=65535
    M=0
fi

##
## k = 20 .. 26
##

# 2<= i <= 8
if [ $i -gt 1 ] && [ $i -lt 9 ]; then
    Ks=$((18+i)) # 20 .. 26
    # 3 .. 2^14-1, ..
    LOW_A=3
    HIGH_A=$(( (1<<(18-2*i))-1 )) # 2^14-1, 2^12-1, .., 2^2-1
    M=0
fi

# 9<= i <= 15
if [ $i -gt 8 ] && [ $i -lt 16 ]; then
    Ks=$((11+i)) # 20 .. 26
    # 2^14+1, .. 2^16-1
    LOW_A=$(( (1<<(32-2*i))+1 )) # 2^14+1, 2^12+1, .., 2^2+1
    HIGH_A=65535
    M=1001
fi

##
## k = 27 .. 32 | p1
##

# 16 <= i <= 21
if [ $i -gt 15 ] && [ $i -lt 22 ]; then
    Ks=$((11+i)) # 27 .. 32
    LOW_A=3
    HIGH_A=$(( (1<<(31-i))-1 )) # 2^15-1, 2^14-1, .., 2^10-1
    M=1001
fi

for K in $Ks; do
    if [ $M -eq 0 ]; then
        echo "$CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results"
        srun $CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results
    else
        echo "$CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results -m $M"
        srun $CODREL -d 1 -k $K -s $LOW_A -S $HIGH_A -f $RESULTS/results -m $M
    fi
done
