#!/bin/bash
#SBATCH --job-name=tfg_nb7
#SBATCH --output=/home/hpc/22231088student/tfg_ids/logs/nb7_%j.log
#SBATCH --error=/home/hpc/22231088student/tfg_ids/logs/nb7_%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00


echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo:   $(hostname)"
echo "Inicio: $(date)"
echo "========================================="


source ~/tfg_env/bin/activate
cd ~/tfg_ids


export PYTHONUNBUFFERED=1


python nb7_escabilidad.py


echo "========================================="
echo "Fin: $(date)"
echo "========================================="



