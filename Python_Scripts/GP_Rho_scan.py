
import os
import sys
import torch
import copy
import subprocess
from pathlib import Path
from pyrokinetics import Pyro, PyroScan
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np
import run_simulations
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Get the root project directory (two levels up)
project_dir = Path(__file__).resolve().parent.parent


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d_3000/"


models = [
            "growth_rate_log", "mode_frequency_log",
        ]


def Read_from_gs2(step_run):

    in_loc = project_dir / "GS2" / "Templates" / step_run / "gs2.in"

    pyro = Pyro(gk_file=in_loc, gk_code="GS2")

    pyro.numerics.nky = 1

    pyro.local_species.electron.domega_drho = 0.0 # this is annoying necessary 

    pyro.local_species.enforce_quasineutrality("electron")

    # Use existing parameter with more realistic ky range
    param_1 = "ky" 
    values_1 = np.arange(0.1, 1, 0.1)/pyro.norms.pyrokinetics.rhoref
    # Add rho parameter with realistic values
    param_2 = "rho"
    values_2 = np.arange(0.2, 1, 0.1)*(pyro.norms.pyrokinetics.lref)
    
    # Dictionary of param and values
    param_dict = {param_1: values_1, param_2: values_2}

    # def enforce_quasineutrality(pyro):
    #     pyro.enforce_quasineutrality()

    # If there are kwargs to function then define here
    param_2_kwargs = {}
    

    pyro_scan_gs2 = PyroScan(
        pyro,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
    )

    # Add proper parameter mapping for beta
    pyro_scan_gs2.add_parameter_key(
        parameter_key="rho",
        parameter_attr="local_geometry", 
        parameter_location=["rho"]
    )

     # Add function to gs2
    # pyro_scan_gs2.add_parameter_func(param_2, enforce_quasineutrality, param_2_kwargs)

    # Create scan directory and write input files
    try:
        pyro_scan_gs2.write(
            file_name="gs2.in",
            base_directory=project_dir / "GS2_Runs" / step_run ,
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None
    
    pyro_copy = copy.copy(pyro)
    # Switch to TGLF
    pyro_copy.gk_code = "TGLF"

     # Create PyroScan object with more descriptive naming
    pyro_scan_tglf = PyroScan(
        pyro_copy,
        param_dict,
        value_fmt=".4f",  # Increased precision for small rho values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
    )
    # Add function to enforce consistent rho prime
    pyro_scan_tglf.add_parameter_key(
        parameter_key="rho",
        parameter_attr="local_geometry", 
        parameter_location=["rho"]
    )

    # Add function to tglf
    # pyro_scan_tglf.add_parameter_func(param_2, enforce_quasineutrality, param_2_kwargs)


    # Create scan directory and write input files
    try:
        pyro_scan_tglf.write(
            file_name="input.tglf",
            base_directory=project_dir / "TGLF_Runs" / step_run ,
            template_file=None
    )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None

    return pyro_scan_tglf,pyro_scan_gs2




def plot_2d(pyro_scan_list,names, Gaussian=False):

    growth_rate_list = []
    mode_freq_list = []

    for pyro_scan in pyro_scan_list:
        pyro_scan.load_gk_output()
        data = pyro_scan.gk_output
        growth_rate_list.append(data["growth_rate"])
        mode_freq_list.append(data["mode_frequency"])
    
    if Gaussian:
        data_gs2_gp = gs2_gp(pyro=pyro_scan_tglf, models_path=models_path, models=models)
        growth_rate_list.append(data_gs2_gp.gk_output["growth_rate_log_M52"])
        mode_freq_list.append(data_gs2_gp.gk_output["mode_frequency_log_M32"])
        names.append("GS2 GP Model")
    
    second_coord_name = list(growth_rate_list[0].coords)[1]


    n_rows = len(growth_rate_list[0].coords[second_coord_name])
    n_cols = 2
    aspect_ratio = 2.0  # width:height ratio per subplot
    width = aspect_ratio * n_cols * 3
    height = n_rows * 2.5
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(len(growth_rate_list[0].coords[second_coord_name]), 2, hspace=0, wspace=0.3)
    axes = np.empty((len(growth_rate_list[0].coords[second_coord_name]), 2), dtype=object)


    for i, second_coor in enumerate(growth_rate_list[0].coords[second_coord_name]):
        # Create subplots
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        axes[i, 0] = ax1
        axes[i, 1] = ax2

        # Plot data
        for growth_rate, mode_frequency, name in zip(growth_rate_list, mode_freq_list, names):
            ax1.plot(growth_rate.ky, growth_rate.sel({second_coord_name: second_coor}).sel(mode=0), label=rf"{name}")
            ax2.plot(mode_frequency.ky, mode_frequency.sel({second_coord_name: second_coor}).sel(mode=0), label=rf"{name}")

        ax1.grid(True)
        ax2.grid(True)

        # Row label on right-hand side
        ax2.text(
            1.05, 0.5,
            rf"{second_coord_name}={second_coor:.2f}",
            transform=ax2.transAxes,
            va='center', ha='left',
            fontsize=10
        )

    for i in range(len(growth_rate_list[0].coords[second_coord_name]) - 1):  # all rows except bottom
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_xlabel("") 
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_xlabel("")

    # Only bottom row gets x-axis labels
    axes[-1, 0].set_xlabel(r"$k_y$")
    axes[-1, 1].set_xlabel(r"$k_y$")



    # Layout and title
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"Plot")

    # Collect all handles and labels from every axis
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

    # Deduplicate by label
    unique = dict(zip(labels, handles))

    # Create one legend for the whole figure
    fig.legend(
        unique.values(),
        unique.keys(),
        loc='upper center',          # position above the subplots
    )

    plt.subplots_adjust(top=0.9, bottom=0.1)  # give room for legend and title


    # Save everything in ONE file
    plt.savefig(f"Plots/plot_test.png", dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    pyro_scan_tglf,pyro_scan_gs2 = Read_from_gs2("SPR-008")
    #run_simulations.tglf_scan(pyro_scan_tglf)
    plot_2d([pyro_scan_tglf],["TGLF"], Gaussian=True)