import re
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

# =======================================================
# Load data
rod_worth_data = pd.read_csv("./rod_worth.csv")
interp_rod_worth = interp1d(rod_worth_data["rod_worth"], rod_worth_data["bank8_pos"])

# =======================================================
# References
rod_worth_ref = rod_worth_data["rod_worth"][399]
beta_ref = np.array([0.0002584, 0.00152, 0.0013908, 0.0030704, 0.001102, 0.0002584])
h_gap_ref = 10000
gamma_frac_ref = 0.019

num_samples = 2000

# =======================================================
# Distributions
dists = {}

# U [max - 0.15 * max, max]
dists["rod_worth"] = stats.uniform(
    loc=rod_worth_ref - 0.15 * rod_worth_ref, scale=0.15 * rod_worth_ref
)

# N [5%]
dists["beta"] = stats.norm(loc=np.sum(beta_ref), scale=np.sum(beta_ref) * 0.05)

# N [20%]
dists["h_gap"] = stats.norm(loc=h_gap_ref, scale=h_gap_ref * 0.2)

# N [20%]
dists["gamma_frac"] = stats.norm(loc=gamma_frac_ref, scale=gamma_frac_ref * 0.2)

# =======================================================
# Samples
for key, value in dists.items():
    dists[key] = value.rvs(size=num_samples)
    plt.hist(dists[key])
    plt.xlabel(key)
    plt.ylabel("Out of 1000 Samples")

# Inputs
inputs = pd.DataFrame(dists)
inputs.to_csv("rea_inputs.csv", index=False)

dists["rod_worth"] = np.insert(dists["rod_worth"], 0, rod_worth_ref)
dists["beta"] = np.insert(dists["beta"], 0, np.sum(beta_ref))
dists["h_gap"] = np.insert(dists["h_gap"], 0, h_gap_ref)
dists["gamma_frac"] = np.insert(dists["gamma_frac"], 0, gamma_frac_ref)


# =======================================================
# Writing inputs files
def write_trans_file(i: int):
    inp = open("C1.inp", "w")

    for line in open("C1B1b1.inp"):
        # Check for new_bank8_pos
        match = re.search("new_bank8_pos", line)

        if match:
            line = re.sub(
                "new_bank8_pos", str(interp_rod_worth(dists["rod_worth"][i])), line
            )

        # Check for new_gamma_frac
        match = re.search("new_gamma_frac", line)

        if match:
            line = re.sub("new_gamma_frac", str(dists["gamma_frac"][i]), line)

        # Check for new_hgap
        match = re.search("new_hgap", line)

        if match:
            line = re.sub("new_hgap", str(dists["h_gap"][i]), line)

        inp.write(line)

    inp.close()


def write_xsec_file(i: int):
    xsec = open("xsec/XSEC_NEACRP", "w")

    beta_fracs = beta_ref / np.sum(beta_ref)

    for line in open("xsec/XSEC_NEACRP_temp"):
        for j in range(6):
            match = re.search("new_beta" + str(j), line)

            if match:
                line = re.sub(
                    "new_beta" + str(j), str(dists["beta"][i] * beta_fracs[j]), line
                )

        xsec.write(line)

    xsec.close()


# =======================================================
# Driver
timing = []
data = []
for i in range(num_samples + 1):
    print("Sample", i)
    # Open input files
    write_trans_file(i)
    write_xsec_file(i)

    # Run PARCS
    start_time = time.time()
    subprocess.run(
        "[PLACE PATH TO PARCS] C1.inp",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    timing.append(time.time() - start_time)
    data.append(pd.read_csv("C1.parcs_plt", delimiter="\\s+", skiprows=[1]).to_numpy())

timing = np.array(timing)
print("Average time per PARCS run = ", np.sum(timing) / timing.size)
data = np.dstack(data)
np.save("outputs3D.npy", data)
