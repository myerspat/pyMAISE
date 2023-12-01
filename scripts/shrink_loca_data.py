# Script that randomly samples input and output data to shrink data
# size from 7486 to 4000 so it can be pushed to GitHub
import pandas as pd
from numpy.random import randint

samples = 4000

# ==================================================================
# Paths
input_path = "./inp_v0.csv"
output_paths = {
    "water_level": "./sv505_v0.csv",
    "core_pressure": "./pn-112A01_v0.csv",
    "break_flow_rate": "./bxmass-255_v0.csv",
    "pct": "./trhmax-900_v0.csv",
}

# ==================================================================
# Input Data
# Read input data
input_data = pd.read_csv(input_path)
print("Input Data\n", input_data)
print(f"Raw Input Shape {input_data.shape}")

# Sample inputs
idxs = randint(low=0, high=input_data.shape[0], size=samples)
sampled_input_data = input_data.iloc[idxs, :]
print(f"Sampled Input Data Shape: {sampled_input_data.shape}")

# Write data
sampled_input_data.to_csv("loca_inp.csv", index=False)

# ==================================================================
# Output Data
for name, path in output_paths.items():
    # Read data
    output = pd.read_csv(path)
    print(f"{name}\n", output)
    print(f"Raw {name} Shape: {output.shape}")

    # Sample data
    sampled_output = output.iloc[:, idxs]
    print(f"Sampled {name} Shape: {sampled_output.shape}")

    # Write data
    sampled_output.to_csv(f"loca_{name}.csv", index=False, header=False)
