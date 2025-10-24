
import os
import sys
import torch
import subprocess
from pathlib import Path
from pyrokinetics import Pyro, PyroScan
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np
import matplotlib.pyplot as plt


# Get the root project directory (two levels up)
project_dir = Path(__file__).resolve().parent.parent


TGLF_BINARY_PATH = os.path.expandvars("/users/hmq514/scratch/TGLF/gacode/tglf/src/tglf")
TGLF_PARSE_SCRIPT = os.path.expandvars("$GACODE_ROOT/tglf/bin/tglf_parse.py")


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d/"


models = [
            "growth_rate_log", "mode_frequency_log",
        ]


def Read_from_gs2(step_run):

    in_loc = project_dir / "GS2" / "Templates" / step_run / "gs2.in"

    pyro = Pyro(gk_file=in_loc, gk_code="GS2")


    pyro.gk_code = "TGLF"

    pyro.numerics.nonlinear = True

    # Use existing parameter with more realistic ky range
    param_1 = "ky" 
    values_1 = np.arange(0.1, 1, 0.1)/pyro.norms.pyrokinetics.rhoref
    
    # Add beta parameter with realistic values
    param_2 = "gamma_exb"
    values_2 = np.arange(0, 0.2, 0.1)*pyro.norms.pyrokinetics.vref/pyro.norms.pyrokinetics.lref
  
    # Dictionary of param and values
    param_dict = {param_1: values_1, param_2: values_2}
  

    # Create PyroScan object with more descriptive naming
    pyro_scan = PyroScan(
        pyro,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
    )

    # Add proper parameter mapping for beta
    pyro_scan.add_parameter_key(
        parameter_key="gamma_exb",
        parameter_attr="numerics", 
        parameter_location=["gamma_exb"]
    )

    # Create scan directory and write input files
    try:
        pyro_scan.write(
            file_name="input.tglf",
            base_directory=project_dir / "TGLF_Runs" / step_run ,
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None

    return pyro_scan


def run_file(sim_dir):
    sim_path = Path(sim_dir)

    # Make sure the directory exists
    sim_path.mkdir(parents=True, exist_ok=True)

    # Parent directory and folder name
    parent = sim_path.parent
    folder = sim_path.name

    # Run the simulation
    subprocess.run(
        ["tglf", "-e", folder],
        cwd=parent
    )

    print(f"Running simulation in directory: {sim_path}")



def run_sim(pyro_scan):
    for run_dir in pyro_scan.run_directories:
        run_file(run_dir)



if __name__ == "__main__":
    pyro_scan_tglf = Read_from_gs2("SPR-045")
    #run_sim(pyro_scan_tglf)
    pyro_scan_tglf.load_gk_output()
    print(pyro_scan_tglf)
    data_tglf = pyro_scan_tglf.gk_output
    print(data_tglf)
    print(data_tglf['ky'])
    # electron_heat_flux_ky = data_tglf["heat"].sel(field="phi", species="electron").isel(time=-1)
    # electron_heat_flux_ky.plot()
    # plot.show()


    