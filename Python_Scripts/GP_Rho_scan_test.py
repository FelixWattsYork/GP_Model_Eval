
import os
import sys
import torch
import copy
import subprocess
from pathlib import Path
from pyrokinetics import Pyro, PyroScan, template_dir
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np
import run_simulations
import logging



# Get the root project directory (two levels up)
project_dir = Path(__file__).resolve().parent.parent


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d_3000/"


models = [
            "growth_rate_log", "mode_frequency_log",
        ]

# Equilibrium file
eq_file = template_dir / "test.geqdsk"

# Kinetics data file
kinetics_file = template_dir / "transp.cdf"




def Read_from_gs2(step_run):

    # Equilibrium file
    eq_file = template_dir / "test.geqdsk"

    # Kinetics data file
    kinetics_file = template_dir / "transp.cdf"

    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        eq_type="GEQDSK",
        kinetics_file=kinetics_file,
        kinetics_type="TRANSP",
    )
    pyro.gk_code = "GS2"

    pyro.numerics.nky = 1
    pyro.numerics.gamma_exb = 0.0
    pyro.local_species.electron.domega_drho = 0.0

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

    return pyro_scan_tglf, pyro_scan_gs2



if __name__ == "__main__":
    pyro_scan_tglf,pyro_scan_gs2 = Read_from_gs2("test")
    run_simulations.tglf_scan(pyro_scan_tglf)
    pyro_scan_tglf.load_gk_output()
    print(pyro_scan_tglf)
    data_tglf = pyro_scan_tglf.gk_output
    print(data_tglf)