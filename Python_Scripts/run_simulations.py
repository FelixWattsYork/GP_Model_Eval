import os
import subprocess
from pathlib import Path
import textwrap
import logging
logger = logging.getLogger(__name__)


GS2_EXE = "/home/Felix/Documents/Physics_Work/Project_Codes/GS2_TGLF/gs2/bin/gs2"


def tglf(run_dir):
    # Make sure the directory exists
    os.makedirs(run_dir, exist_ok=True)

    parent = os.path.dirname(run_dir)
    folder = os.path.basename(run_dir)

    subprocess.run(
        ["tglf", "-e", folder],
        cwd=parent
    )
    logger.info(f"Running simulation in directory: {run_dir}")

def gs2_viking(run_dir):
    run_dir = Path(run_dir)
    job_name = f"gs2_{run_dir.name}"
    logger.info(f"Creating GS2 Slurm Script in directory: {run_dir}")
    slurm_script = textwrap.dedent(f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}            # Job name
#SBATCH --partition=nodes               # What partition the job should run on
#SBATCH --time=0-32:00:00               # Time limit (DD-HH:MM:SS)
#SBATCH --ntasks=96                      # Number of MPI tasks to request
#SBATCH --cpus-per-task=1               # Number of CPU cores per MPI task
#SBATCH --exclusive                     # tries to take the core exclusively
#SBATCH --account=pet-gspt-2019         # Project account to use
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hmq514@york.ac.uk   # Where to send mail
#SBATCH --output={job_name}-%j.log              # Standard output log
#SBATCH --error={job_name}-%j.err              # Standard error log

# Purge any previously loaded modules
module purge

# Load modules
module load gompi/2022b OpenMPI/4.1.4-GCC-12.2.0 netCDF-Fortran/4.6.0-gompi-2022b FFTW/3.3.10-GCC-12.2.0 OpenBLAS/0.3.21-GCC-12.2.0 Python/3.10.8-GCCcore-12.2.0

# Commands to run. 

export GK_SYSTEM='viking'
export MAKEFLAGS='-IMakefiles'
export HDF5_USE_FILE_LOCKING=FALSE

ulimit -s unlimited

######################## Above is in bashrc anyway

export OMP_NUM_THREADS=1
INPUT_DIR=$(dirname "{run_dir}")  # Extract directory of input file
srun --hint=nomultithread --distribution=block:block -n 96 {GS2_EXE} {run_dir}/gs2.in | tee OUTPUT
    """)

    # Write the Slurm script to the run directory
    script_path = run_dir / "jobscript.job"
    script_path.write_text(slurm_script)

    # Submit with sbatch
    logger.info(f"Running gs2 simulation in directory: {run_dir}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)

    # Extract job ID from sbatch output
    logger.info(f"gs2 simulation job ID: {result.stdout.strip()}")




def gs2_Local(run_dir):
    run_dir = Path(run_dir)
    logger.info(f"running gs2 simulation in directory: {run_dir}")
    cmd = [
    "mpirun", "-n", "4",
    GS2_EXE,
    run_dir / "gs2.in"
    ]
    # Run and wait for it to finish
    result = subprocess.run(cmd, capture_output=True, text=True)


def tglf_scan(pyro_scan):
    for run_dir in pyro_scan.run_directories:
        tglf(run_dir)


def gs2_scan(pyro_scan):
    if os.getenv("GACODE_PLATFORM") == "VIKING":
        for run_dir in pyro_scan.run_directories:
            gs2_viking(run_dir)
    elif os.getenv("GACODE_PLATFORM") == "ARCH":
        for run_dir in pyro_scan.run_directories:
            gs2_Local(run_dir)
    else:
        print("NO VALID PLAFORM TO RUN GS2 ON!")

