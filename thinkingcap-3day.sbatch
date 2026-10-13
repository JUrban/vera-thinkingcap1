#!/usr/bin/env bash
#SBATCH -A REPLACE_WITH_YOUR_VERA_PROJECT
#SBATCH -p vera
#SBATCH --job-name=thinkingcap-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=A40:1
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=thinkingcap-%j.log

set -euo pipefail
# Submit from a persistent directory, such as ~/vera-thinkingcap.
# Must authenticate to Hugging Face once before running this batch job.
srun "$HOME/vera-thinkingcap/start-thinkingcap.sh"
