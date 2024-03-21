#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:21:37 2024

@author: jacobcooper
"""

import numpy as np
import pandas as pd
import xarray as xr

def load_anomaly(path, stack_series, multiclass):
    """
    Process the raw data for binary or multiclass classification.

    Args:
    - path (str): Path to the raw data file.
    - stack_series (bool): Whether the pulses should be stacked (3D or 2D).
    - multiclass (bool): Whether to return the binary or multiclass classification.

    Returns:
    - X_dataarray (xarray.DataArray): The processed input data as a DataArray.
    - Y_dataarray (xarray.DataArray): The processed labels for classification as a DataArray.
    """
    system = 'DTL'  # Choose the system to load and process data for: RFQ, DTL, CCL, or SCL

    # Load the data
    X = np.load('%s.npy' % system)
    Y = np.load('%s_labels.npy' % system, allow_pickle=True)

    Y = Y[:, 1:]  # Remove Filepath
    if multiclass:
        Y_df = pd.DataFrame(Y, columns=['state', 'type'])
        Y_processed = Y_df.to_numpy()
        Y_processed = np.where(Y_processed[:, 0] == 'Run', 1, 0)
        # Relabeling script
        Y_relab = Y_df.copy()
        for i in range(Y_df.shape[0]):
            if Y_df.iloc[i, 1] == 'Normal':
                pass
            elif 'IGBT' in Y_df.iloc[i, 1]:
                Y_relab.iloc[i, 1] = 'IGBT Fault'
            elif 'CB' in Y_df.iloc[i, 1] or 'CapBank' in Y_df.iloc[i, 1] or 'TPS' in Y_df.iloc[i, 1]:
                Y_relab.iloc[i, 1] = 'CB or TPS Fault'
            elif 'Driver' in Y_df.iloc[i, 1]:
                Y_relab.iloc[i, 1] = 'Driver Fault'
            elif 'FLUX' in Y_df.iloc[i, 1] or 'Flux' in Y_df.iloc[i, 1]:
                Y_relab.iloc[i, 1] = 'Flux Fault'
            elif 'DV/DT' in Y_df.iloc[i, 1]:
                Y_relab.iloc[i, 1] = 'DV/DT Fault'
            elif 'SCR' in Y_df.iloc[i, 1]:
                Y_relab.iloc[i, 1] = 'SCR AC Input Fault'
        for i in range(Y_relab.shape[0]):
            if Y_relab.iloc[i, 1] not in ['Normal', 'Fiber Fault', 'DV/DT Fault', 'SCR AC Input Fault',
                                          'Driver Fault', 'SNS PPS Missing', 'Flux Fault', 'CB or TPS Fault']:
                Y_relab.iloc[i, 1] = 'Misc/Unknown'
        Y_processed = Y_relab.to_numpy()
        Y_processed = Y_processed[:, 1:]

    else:
        # Binary classification: Convert 'Run' to 1 and 'Fault' to 0
        Y_processed = Y[:, :-1]
        Y_processed[:, 0] = (np.where(Y_processed[:, 0] == 'Run', 1, 0)).astype(int)  # Change run/fail str to 1/0 int

    # If stack_series is False, reshape the arrays to 2D
    if stack_series:
        Y_processed = np.repeat(Y_processed[:, np.newaxis, :], X.shape[1], axis=1)
    else:
        Y_expanded = np.repeat(Y_processed[:, np.newaxis, :], X.shape[1], axis=1)
        # Reshape (Y_expanded) to (samples * timesteps, features)
        Y_processed = Y_expanded.reshape(-1, Y_expanded.shape[2])
        X = X.reshape(-1, X.shape[2])

    # Convert numpy arrays to xarray DataArrays
    X_dataarray = xr.DataArray(X)
    Y_dataarray = xr.DataArray(Y_processed)

    print("X shape:", X.shape)
    print("Y_processed shape:", Y_processed.shape)

    # Return the processed input data and labels as DataArrays
    return X_dataarray, Y_dataarray

# Example usage:
path = '/Users/jacobcooper/Desktop/Real Electronic Signal Data from Particle Accelerator Power Systems for Machine Learning Anomaly Detection/DTL'

# Set stack_series to False if you want to reshape the data into 2D
# Set multiclass to True if you want multiclass labels
X_dataarray, Y_dataarray = load_anomaly(path, stack_series=True, multiclass=True)
#print("X_dataarray:", X_dataarray)
#print("Y_dataarray:", Y_dataarray)
print("X_dataarray:", X_dataarray.shape)
print("Y_dataarray:", Y_dataarray)
