from pathlib import Path
from pyrokinetics import Pyro, PyroScan
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
from pyrokinetics.diagnostics.saturation_rules import SaturationRules
import numpy as np
import matplotlib.pyplot as plt
from pyrokinetics import Pyro,PyroScan
from pyrocomparison import PyroComparison
import numpy as np
import run_simulations

project_dir = Path(__file__).resolve().parent.parent

# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d/"


models = [
            "growth_rate_log", "mode_frequency_log", "kperp2_phi_log", "kperp2_apa_log",
            "kperp2_bpar_log", "totIonFlux_log", "totElecFlux_log", "totPartFlux_log",
            "apa_phi_log", "bpar_phi_log"
        ]

# Choose convention
output_convention = "pyrokinetics"

# Base file
base_dir = "/home/Felix/Documents/Physics_Work/Project_Codes/GP_Model_Eval/GS2/Templates/SPR-045"
base_file = "gs2.in"

pyro = Pyro(gk_file=f"{base_dir}/{base_file}")


base_ky = pyro.numerics.ky.to(pyro.norms.pyrokinetics) / 2

# Set up ky and theta0 grid
param_1 = "ky"
param_2 = "th0"

kys = np.array([2, 3, 5, 10, 20, 30, 40, 50, 70, 100, 120, 140]) * base_ky
th0s = np.array([0, 0.1, 0.2, 0.4, 1.2, 3.14])

values_1 = kys
values_2 = th0s


# Dictionary of param and values
param_dict = {param_1: values_1, param_2: values_2}

# Create PyroScan object
pyro_scan = PyroScan(
    pyro,
    param_dict,
    value_fmt=".4f",  # Increased precision for small beta values
    value_separator="_",
    parameter_separator="_",
)


# Add in path to each defined parameter to scan through
pyro_scan.add_parameter_key(
    param_1, "gk_input", ["data", "kt_grids_single_parameters", "n0"]
)

try:
    pyro_scan.write(
        file_name="input.tglf",
        base_directory=project_dir / "GS2_Runs" / "SPR-045" ,
        template_file=None
    )

except Exception as e:
    print(f"Error writing parameter scan files: {e}")

run_simulations.gs2_scan(pyro_scan)

# Load outputs
pyro_scan.load_gk_output(output_convention=output_convention, tolerance_time_range=0.9)

# Create saturation object
saturation = SaturationRules(pyro_scan)

# Inputs for QL model
alpha = 2.5
Q0 = 25

# Must match convention
gamma_exb = (
    0.04380304982261718 * pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref
)

gk_output = saturation.mg_saturation(
    Q0=Q0,
    alpha=alpha,
    gamma_exb=gamma_exb,
    output_convention=output_convention,
    gamma_tolerance=0.3,
    theta0_dim="th0",
)

print("GK output", gk_output)
print("Heat flux calculation", gk_output["heat"])
print("Gamma flux calculation", gk_output["particle"])