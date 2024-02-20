import numpy as np
import pandas as pd

outputs3D = np.load("outputs3D.npy")

# ======================================
# Collected outputs
max_fuel_temp = []
max_power = []
burst_width = []
avg_outlet_temp = []

# ======================================
# Get max and averages
for i in range(outputs3D.shape[2]):
    max_fuel_temp.append(np.max(outputs3D[:, 4, i]))
    max_power.append(np.max(outputs3D[:, 2, i]))
    avg_outlet_temp.append(np.average(outputs3D[:, 5, i]))

# ======================================
# Calculate burst width
for i in range(outputs3D.shape[2]):
    dist = outputs3D[:, 2, i] - 0.0001

    max_p = np.max(dist)
    position = np.argmax(dist)

    left_pos = np.abs(dist[:position] - max_p / 2).argmin()
    right_pos = np.abs(dist[position:] - max_p / 2).argmin()

    burst_width.append(outputs3D[left_pos, 0, i] - outputs3D[right_pos, 0, i])

pd.DataFrame(
    {
        "max_power": max_power,
        "burst_width": burst_width,
        "max_Tf": max_fuel_temp,
        "avg_Tcool": avg_outlet_temp,
    }
).to_csv("rea_outputs.csv", index=False)
