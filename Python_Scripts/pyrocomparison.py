from typing import Dict, List, Any, Optional
from pyrokinetics import Pyro,PyroScan
import numpy as np
import torch
from pathlib import Path
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import run_simulations


FILE_NAME_DICT = {"GS2":"gs2.in","TGLF":"input.tglf"}


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d/"



models = [
            "growth_rate_log", "mode_frequency_log",
        ]

class PyroComparison:
    """
    Compare Pyrokinetic scans (e.g. GS2, TGLF, CGYRO) using a common input file.
    Supports multiple codes, per-code parameter variations, and per-run input files.
    """

    def __init__(self,output_loc, input_file: str, param_dict,param_keys, codes: Dict[str, Optional[List[Dict[str, Any]]]]):
        """
        Args:
            input_file (str): Global input file (used unless overridden per run).
            codes (dict): Mapping of code name to optional list of variation dicts.
                Example:
                    {
                        "gs2": None,  # Single default run
                        "tglf": [
                            {"name": "tglf_gamma_high", "flags": {"GAMMA_OPT": 2}},
                            {"name": "tglf_gamma_low", "flags": {"GAMMA_OPT": 1}},
                            {"name": "tglf_custom_input", "flags": {"GAMMA_OPT": 2}, "input_file": "tglf_alt.in"}
                        ]
                    }
        """
        self.output_loc = Path(output_loc)
        self.global_input_file = Path(input_file)
        self.param_keys = param_keys
        self.codes = codes
        self.param_dict = param_dict
        self.runs = self._prepare_runs()

    def _prepare_runs(self) -> List[Dict[str, Any]]:
        """
        Build a list of runs with proper names and input file handling.
        """
        runs = []
        for code, variations in self.codes.items():
            if not variations:
                pyro = Pyro(gk_file=self.global_input_file)
                pyro.gk_code = code
                # Create PyroScan object with more descriptive naming
                pyro_scan = PyroScan(
                    pyro,               
                    self.param_dict,
                    value_fmt=".4f",  # Increased precision for small beta values
                    value_separator="_",
                    parameter_separator="_",
                )
                for key in self.param_keys:
                    print(key)
                    pyro_scan.add_parameter_key(
                        parameter_key=key[0],
                        parameter_attr=key[1], 
                        parameter_location=key[2]
                    )

                # Default run for this code
                runs.append({
                    "pyroscan": pyro_scan,
                    "name": code,  # Default name = code name
                    "flags": {},
                    "input_file": self.global_input_file
                })

            else:
                for var in variations:
                    print("got here")
                    pyro = Pyro(gk_file=var.get("input_file", self.global_input_file))
                    pyro.gk_code = code
                    pyro.gk_input.add_flags(var.get("new_flags", {}))
                    flag_to_set = var.get("flags", {})
                    print(flag_to_set)
                    for key, value in flag_to_set.items():
                        pyro.gk_input.data[key] = value
                    pyro_scan = PyroScan(
                        pyro,               
                        self.param_dict,
                        value_fmt=".4f",  # Increased precision for small beta values
                        value_separator="_",
                        parameter_separator="_",
                    )
                    for key in self.param_keys:
                        pyro_scan.add_parameter_key(
                            parameter_key=key[0],
                            parameter_attr=key[1], 
                            parameter_location=key[2]
                        )
                    runs.append({
                        "pyroscan": pyro_scan,
                        "name": var.get("name", code),  # Default to code name
                        "flags": var.get("flags", {}),
                        "input_file": var.get("input_file", self.global_input_file)
                    })
        for run in runs:
            try:
                print(run["pyroscan"].parameter_dict)
                run["pyroscan"].write(
                    file_name= FILE_NAME_DICT[code],
                    base_directory= self.output_loc / f"{run["name"]}_Runs",
                    template_file=None
                )
            except Exception as e:
                print(f"Error writing parameter scan files: {e}")
                return None
        return runs


    def run_all(self):
        for run in self.runs:
            if run["pyroscan"].gk_code=="TGLF":
                run_simulations.tglf_scan(run["pyroscan"])
            elif run["pyroscan"].gk_code=="GS2":
                run_simulations.gs2_scan(run["pyroscan"])


    def plot_2d(self, Gaussian=False):

        growth_rate_list = []
        mode_freq_list = []
        names = []

        for run in self.runs:
            run["pyroscan"].load_gk_output()
            data = run["pyroscan"].gk_output
            growth_rate_list.append(data["growth_rate"])
            mode_freq_list.append(data["mode_frequency"])
            names.append(run["name"])
        
        if Gaussian:
            data_gs2_gp = gs2_gp(pyro=self.runs[-1]["pyroscan"], models_path=models_path, models=models)
            growth_rate_list.append(data_gs2_gp.gk_output["growth_rate_log_M12"])
            mode_freq_list.append(data_gs2_gp.gk_output["mode_frequency_log_M12"])
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