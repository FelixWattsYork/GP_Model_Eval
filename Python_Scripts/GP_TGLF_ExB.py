
import os
import sys
import torch
import subprocess
from pathlib import Path
from pyrokinetics import Pyro, PyroScan
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent   # repo root if this file sits at repo root
STEP_DATA_DIR = "/home/Felix/Documents/Physics_Work/Project_Codes/Beta_Prime_Scan/gs2_gamma_exb_scan"
STEP_CASE = "n40"

TGLF_BINARY_PATH = os.path.expandvars("/users/hmq514/scratch/TGLF/gacode/tglf/src/tglf")
TGLF_PARSE_SCRIPT = os.path.expandvars("$GACODE_ROOT/tglf/bin/tglf_parse.py")


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d_3000/"


models = [
            "growth_rate_log", "mode_frequency_log",
        ]


def Read_from_gs2():
    in_loc = f"{STEP_DATA_DIR}/{STEP_CASE}/gs2.in"
    pyro = Pyro(gk_file=in_loc, gk_code="GS2")

    pyro.numerics.nky = 1
    pyro.numerics.gamma_exb = 0.0
    pyro.local_species.electron.domega_drho = 0.0

    # Use existing parameter with more realistic ky range
    param_1 = "ky" 
    values_1 = np.arange(0.1, 1, 0.1)/pyro.norms.pyrokinetics.rhoref
    # Add gamma_exb parameter with realistic values
    param_2 = "gamma_exb"
    values_2 = np.arange(0.01, 1, 0.01)*(pyro.norms.pyrokinetics.vref/pyro.norms.pyrokinetics.lref)
    
    # Dictionary of param and values
    param_dict = {param_1: values_1, param_2: values_2}

    # Switch to TGLF
    pyro.gk_code = "TGLF"

     # Create PyroScan object with more descriptive naming
    pyro_scan_tglf = PyroScan(
        pyro,
        param_dict,
        value_fmt=".4f",  # Increased precision for small gamma_exb values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
    )
    # Add function to enforce consistent gamma_exb prime
    pyro_scan_tglf.add_parameter_key(
        parameter_key="gamma_exb",
        parameter_attr="numerics", 
        parameter_location=["gamma_exb"]
    )

    # Create scan directory and write input files
    #try:
    pyro_scan_tglf.write(
        file_name="input.tglf",
        base_directory=REPO_ROOT / "parameter_scan_exb_tglf",
        template_file=None
    )
    # except Exception as e:
    #     print(f"Error writing parameter scan files: {e}")
    #     return None

    return pyro_scan_tglf

def run_file_full(sim_dir):
    # Make sure the directory exists
    os.makedirs(sim_dir, exist_ok=True)

    parent = os.path.dirname(sim_dir)
    folder = os.path.basename(sim_dir)

    subprocess.run(
        ["tglf", "-e", folder],
        cwd=parent
    )
    print(f"Running simulation in directory: {sim_dir}")

def run_sim(pyro_scan):
    for run_dir in pyro_scan.run_directories:
        run_file_full(run_dir)



def load_results(pyro_scan_tglf):
    # Load output from tglf
    pyro_scan_tglf.load_gk_output()

    data_tglf = pyro_scan_tglf.gk_output
    growth_rate_tglf = data_tglf['growth_rate']
    print(growth_rate_tglf)
    print("tglf data")
    mode_frequency_tglf = data_tglf['mode_frequency']
    print(mode_frequency_tglf)


    # load GS2_GP results
    #need to create a function to load a gs2_gp for a pyrosca

    data_gs2_gp = gs2_gp(pyro=pyro_scan_tglf, models_path=models_path, models=models)
    
    growth_rate_gs2_gp = data_gs2_gp.gk_output["growth_rate_log_M52"]
    mode_frequency_gs2_gp = data_gs2_gp.gk_output["mode_frequency_log_M32"]
    print("gs2_gp data")
    print(growth_rate_gs2_gp)
    print(f"growth rate: {growth_rate_gs2_gp.ky}")
    print(f"growth rate: {growth_rate_gs2_gp.gamma_exb}")
    print(mode_frequency_gs2_gp)


    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Plot growth rate and mode frequency vs ky for different gamma_exb values

    
    
    fig = plt.figure(figsize=(9, 3*len(growth_rate_tglf.gamma_exb)))
    gs = gridspec.GridSpec(len(growth_rate_tglf.gamma_exb), 2, hspace=0, wspace=0.3)

    axes = np.empty((len(growth_rate_tglf.gamma_exb), 2), dtype=object)

    for i, gamma_exb in enumerate(growth_rate_tglf.gamma_exb.values):
        # Create subplots
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        axes[i, 0] = ax1
        axes[i, 1] = ax2

        # Plot data
        ax1.plot(growth_rate_tglf.ky, growth_rate_tglf.sel(gamma_exb=gamma_exb).sel(mode=0), label=rf"tglf_$\gamma_exb={gamma_exb:.3f}$")
        ax1.plot(growth_rate_gs2_gp.ky, growth_rate_gs2_gp.sel(gamma_exb=gamma_exb,output="value"), label=rf"GS2_$\gamma_exb={gamma_exb:.3f}$")
        ax2.plot(mode_frequency_tglf.ky, mode_frequency_tglf.sel(gamma_exb=gamma_exb).sel(mode=0), label=rf"tglf_$\gamma_exb={gamma_exb:.3f}$")
        ax2.plot(mode_frequency_gs2_gp.ky, mode_frequency_gs2_gp.sel(gamma_exb=gamma_exb,output="value"), label=rf"GS2_$\gamma_exb={gamma_exb:.3f}$")

        # Axis labels
        ax1.set_ylabel(r'$\gamma (c_{s}/a)$')
        ax2.set_ylabel(r'$\omega (c_{s}/a)$')

        ax1.grid(True)
        ax2.grid(True)

        # Row label on right-hand side
        ax2.text(
            1.05, 0.5,
            rf"$\gamma_exb={gamma_exb:.2f}$",
            transform=ax2.transAxes,
            va='center', ha='left',
            fontsize=10
        )
    for i in range(len(growth_rate_tglf.gamma_exb) - 1):  # all rows except bottom
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_xlabel("") 
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_xlabel("")
    # Only bottom row gets x-axis labels
    axes[-1, 0].set_xlabel(r"$k_y$")
    axes[-1, 1].set_xlabel(r"$k_y$")

    # Layout and title
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"Plot of Growth rate and Frequency against $k_y$ for different $\gamma_exb$ or STEP Case SPR-045",
                fontsize=16, y=0.95)
    fig.legend()

    # Save everything in ONE file
    file_path_str = f"ExB_Scans/{STEP_CASE}_all_gamma_exbs_pairs.png"

    # 1. Convert the path string to a Path object
    save_path = Path(file_path_str)

    # 2. Extract the parent directory
    # The .parent attribute gives the directory path
    save_dir = save_path.parent

    # 3. Create the directory if it doesn't exist
    # parents=True creates any necessary parent directories
    # exist_ok=True prevents an error if the directory already exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save the figure using the original path string or the Path object
    plt.savefig(file_path_str, dpi=300)
    plt.close(fig)




    # Plot growth rate and mode frequency vs gamma_exb for different ky values

    fig = plt.figure(figsize=(9, 3*len(growth_rate_tglf.ky)))
    gs = gridspec.GridSpec(len(growth_rate_tglf.ky), 2, hspace=0, wspace=0.3)

    axes = np.empty((len(growth_rate_tglf.ky), 2), dtype=object)

    for i, ky in enumerate(growth_rate_tglf.ky.values):
        print(f"ky is {ky}")
        # Create subplots run_sim(pyro_scan)
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        axes[i, 0] = ax1
        axes[i, 1] = ax2

        # Plot data
        ax1.plot(growth_rate_tglf.gamma_exb, growth_rate_tglf.sel(ky=ky).sel(mode=0), label=rf"tglf_$k_y={ky:.2f}$")
        ax1.plot(growth_rate_gs2_gp.gamma_exb, growth_rate_gs2_gp.sel(ky=ky,output="value"), label=rf"gs2_$k_y={ky:.2f}$")
        ax2.plot(mode_frequency_tglf.gamma_exb, mode_frequency_tglf.sel(ky=ky).sel(mode=0), label=rf"tglf_$k_y={ky:.2f}$")
        ax2.plot(mode_frequency_gs2_gp.gamma_exb, mode_frequency_gs2_gp.sel(ky=ky,output="value"), label=rf"gs2_$k_y={ky:.2f}$")

        # Axis labels
        ax1.set_ylabel(r'$\gamma (c_{s}/a)$')
        ax2.set_ylabel(r'$\omega (c_{s}/a)$')

        ax1.grid(True)
        ax2.grid(True)

        # Row label on right-hand side
        ax2.text(
            1.05, 0.5,
            rf"$k_y={ky:.2f}$",
            transform=ax2.transAxes,
            va='center', ha='left',
            fontsize=10
        )
    for i in range(len(growth_rate_tglf.ky) - 1):  # all rows except bottom
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_xlabel("") 
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_xlabel("")
    # Only bottom row gets x-axis labels
    axes[-1, 0].set_xlabel(r"$\gamma_exb$")
    axes[-1, 1].set_xlabel(r"$\gamma_exb$")

    # Layout and title
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"Plot of Growth rate and Frequency against $\gamma_exb$ for different $k_y$ for STEP Case SPR-045",
                fontsize=16, y=0.95)
    fig.legend()

    # Save everything in ONE file
    plt.savefig(f"ExB_Scans/{STEP_CASE}_all_ky_pairs.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    pyro_scan_tglf = Read_from_gs2()
    run_sim(pyro_scan_tglf)
    load_results(pyro_scan_tglf)