import copy

import numpy as np
import pandas as pd
import xarray as xr
from pkg_resources import resource_filename

from pyMAISE.preprocessing import read_csv


def _get_full_path(path: str):
    """Get full pyMAISE data file path"""
    return resource_filename("pyMAISE", path)


def load_MITR():
    """
    Load MIT reactor data. There are 6 inputs, control blade height :math:`[cm]` (labeled as ``CB#``),
    and 22 outputs (labeled as ``A-#``, ``B-#``, or ``C-#``), fuel element power :math:`[W]`. This
    data comes from :cite:`RADAIDEH2023112423` and was constructed through the perturbation of
    the control blade heights and the simulation of the reactor in MCNP to determine the expected
    power in each element. This data set includes 1000 samples.

    Returns
    -------
    data: xarray.DataArray
        Raw MIT reactor data.
    inputs: xarray.DataArray
        Control blade heights.
    outputs: xarray.DataArray
        Fuel element power.
    """
    return read_csv(
        [_get_full_path("datasets/crx.csv"), _get_full_path("datasets/powery.csv")],
    )


def load_xs():
    """
    Load reactor physics data. There are 1000 samples with 8 cross sections (XS)
    :math:`[cm^{-1}]` as inputs:

    - ``FissionFast``: fast fission,
    - ``CaptureFast``: fast capture,
    - ``FissionThermal``: thermal fission,
    - ``CaptureThermal``: thermal capture,
    - ``Scatter12``: group 1 to 2 scattering,
    - ``Scatter11``: group 1 to 1 scattering,
    - ``Scatter21``: group 2 to 1 scattering,
    - ``Scatter22``: group 2 to 2 scattering,

    with output of :math:`k`, the neutron multiplication factor. This data
    was taken from :cite:`RADAIDEH2019264`, a sensitivity analysis using
    the Shapley effect. The geometry of the problem is a pressurized water
    reactor (PWR) lattice based on the BEAVRS benchmark. The lattice utilizes
    quarter core symmetry in TRITON and is depleted to :math:`50~GWD/MTU`.
    The data was constructed using a two step process:

    1. the uncertainty in the fundamental microscopic XS data was propogated,
    2. and these XSs were collapsed into a 2-group form using

    .. math::
        \Sigma_x^g = \\frac{\int_{\Delta E_g}dE\int_V\Sigma_{x, m}(E)\phi(r, E, t)dV}
        {\int_{\Delta E_g}dE\int_V\phi(r, E, t)dV}.

    The Sampler module in SCALE was used for uncertainty propogation, and the
    56-group XS and covariance libraries were used in TRITON to create 56-group
    homogeneous XSs using the above equation. The homogeneous XSs were then collapsed
    into a 2-group library. 1000 random samples were taken from the Sampler.

    Returns
    -------
    data: xarray.DataArray
        Raw reactor physics data.
    inputs: xarray.DataArray
        Cross sections.
    outputs: xarray.DataArray
        :math:`k`, neutron multiplication factor, data.
    """
    return read_csv(_get_full_path("datasets/xs.csv"), slice(0, -1), slice(-1, None))


def load_fp():
    """
    Load fuel performance data. This data set consists of 13 inputs:

    - ``fuel_dens``: fuel density :math:`[kg/m^3]`,
    - ``porosity``: porosity,
    - ``clad_thick``: cladding thickness :math:`[m]`,
    - ``pellet_OD``: pellet outer diameter :math:`[m]`,
    - ``pellet_h``: pellet height :math:`[m]`,
    - ``gap_thickness``: gap thickness :math:`[m]`,
    - ``inlet_T``: inlet temperature :math:`[K]`,
    - ``enrich``: U-235 enrichment,
    - ``rough_fuel``: fuel roughness :math:`[m]`,
    - ``rough_clad``: cladding roughness :math:`[m]`,
    - ``ax_pow``: axial power,
    - ``clad_T``: cladding surface temperature :math:`[K]`,
    - ``pressure``: pressure :math:`[Pa]`,

    and 4 outputs:

    - ``fis_gas_produced``: fission gas production :math:`[mol]`,
    - ``max_fuel_centerline_temp``: max fuel centerline temperature :math:`[K]`,
    - ``max_fuel_surface_temperature``: max fuel surface temperature :math:`[K]`,
    - ``radial_clad_dia``: radial cladding diameter displacement after irradiation :math:`[m]`,

    with 400 data points. This data is case 1 from :cite:`RADAIDEH2020106731` which is
    based on the pellet-cladding mechanical interaction (PCMI) benchmark. The 13 inputs were
    uniformly randomly sampled independently within their uncertainty bounds and simulated
    in BISON. The rod response was recorded in 4 outputs.

    Returns
    -------
    data: xarray.DataArray
        Raw fuel performance data.
    inputs: xarray.DataArray
        13 inputs.
    outputs: xarray.DataArray
        4 outputs.
    """
    return read_csv(
        [_get_full_path("datasets/fp_inp.csv"), _get_full_path("datasets/fp_out.csv")]
    )


def load_heat():
    """
    Load the heat conduction data. This data consists of 1000 samples of 7 inputs:

    - ``qprime``: linear heat generation rate :math:`[W/m]`,
    - ``mdot``: mass flow rate :math:`[g/s]`,
    - ``Tin``: temperature of the fuel boundary :math:`[K]`,
    - ``R``: fuel radius :math:`[m]`,
    - ``L``: fuel length :math:`[m]`,
    - ``Cp``: heat capacity :math:`[J/(g\cdot K)]`,
    - ``k``: thermal conductivity :math:`[W/(m\cdot K)]`,

    with 1 output:

    - ``T``: fuel centerline temperature :math:`[K]`.

    The data set was constructed through Latin hypercube sampling of the 7 input
    parameters for heat conduction through a fuel rod. These samples were then
    used to solve for the fuel centerline temperature analytically. We assume
    volumetric heat generation is uniform radially. The problem is defined by

    .. math::
        \\frac{1}{r}\\frac{d}{dr}\Big(kr\\frac{dT}{dr}\Big) + q''' = 0

    with two boundary conditions: :math:`\\frac{dT}{dr}\Big|_{r=0}=0` and
    :math:`T(R) = T_{in}`. Therefore, the temperature profile in the fuel is

    .. math::
        T(r) = \\frac{q'}{4\pi k}(1 - (r/R)^2) + T_{in}.

    Returns
    -------
    data: xarray.DataArray
        Raw heat conduction data.
    inputs: xarray.DataArray
        7 inputs.
    outputs: xarray.DataArray
        Fuel centerline temperature.
    """
    return read_csv(_get_full_path("datasets/heat.csv"), slice(0, -1), slice(-1, None))


def load_rea():
    """
    Load NEACRP C1 rod ejection accident (REA) data. This data consists of 2000 samples of 4 inputs:

    - ``rod_worth``: reactivity worth of the ejected rod,
    - ``beta``: delayed neutron fraction,
    - ``h_gap``: gap conductance :math:`[W/(m^2\cdot K)]`,
    - ``gamma_frac``: direct heating fraction,

    with 4 outputs:

    - ``max_power``: peak power reached during transient :math:`[\%FP]`,
    - ``burst_width``: Width of power burst :math:`[s]`,
    - ``max_TF``: max fuel centerline temperature :math:`[K]`,
    - ``avg_Tcool``: average coolant outlet temperature :math:`[K]`.

    This data set was constructed by perturbing the inputs listed above prior to REA transient
    simulated in PARCS.

    Returns
    -------
    data: xarray.DataArray
        Raw rod ejection data.
    inputs: xarray.DataArray
        4 inputs.
    outputs: xarray.DataArray
        4 outputs.
    """
    return read_csv(
        [_get_full_path("datasets/rea_inputs.csv"), _get_full_path("datasets/rea_outputs.csv")],
    )


# # Load BWR micro-reactor data
# def load_BWR():
#     preprocessor = PreProcessor()
#     preprocessor.read_csv(
#         [get_full_path("data/bwr_input.csv"), get_full_path("data/bwr_output.csv")],
#     )
#     return preprocessor
#
#
# # Load HTGR micro reactor quadrant power data before preprocessing
# def load_qpower():
#     preprocessor = PreProcessor()
#     preprocessor.read_csv(
#         get_full_path("data/microreactor.csv"), slice(29, 37), slice(4, 8)
#     )
#     return preprocessor
#
#
# # Load HTGR micro-reactor quadrant power data after preprocessing using symmetry conditions
# def load_pqpower():
#     preprocessor = PreProcessor()
#     preprocessor.read_csv(
#         get_full_path("data/microreactor_preprocessed.csv"), slice(1, 9), slice(9, 14)
#     )
#     return preprocessor


# Load and prep LOCA data
def load_loca(stack_series=False):
    """
    Load loss of coolant accident (LOCA) time series data. This data comes from
    :cite:`RADAIDEH2020113699` and consists of 40 time-independent features that describe
    the initial state of the reactor during the LOCA transient which are propogated out
    in time. There are 4 sequences:

    - ``Pellet Cladding Temperature``: pellet cladding temperature [K],
    - ``Core Pressure``: core pressure [Pa],
    - ``Water Level``: water level [m],
    - ``Break Flow Rate``: break flow rate [kg/s],

    with 400 time steps.

    Parameters
    ----------
    stack_series: bool, default=False
        If false, the data is loaded in 3D with dimensions (samples, time
        steps, features). If true, the data is loaded in 2D with the
        sequences stacked like pulse data resulting in dimensions
        (time steps, features).

    Returns
    -------
    nominal_data: xarray.DataArray
        The 2D or 3D nominal LOCA data. If 2D it will be shape (400, 44)
        and if 3D then the shape is (1, 400, 44).
    perturbed_data: xarray.DataArray
        The 2D or 3D perturbed LOCA data. If 2D it is shape (1600000, 44)
        and if 3D then the shape is (4000, 400, 44).

    """
    # Paths
    input_path = _get_full_path("datasets/loca_inp.csv")
    output_paths = {
        "Pellet Cladding Temperature": "datasets/loca_pct.csv",
        "Core Pressure": "datasets/loca_core_pressure.csv",
        "Water Level": "datasets/loca_water_level.csv",
        "Break Flow Rate": "datasets/loca_break_flow_rate.csv",
    }

    # Read outputs and concatenate arrays
    outputs = []
    for path in output_paths.values():
        outputs.append(
            pd.read_csv(_get_full_path(path), header=None).values.T[:, :, np.newaxis]
        )
    outputs = np.concatenate(outputs, axis=-1)

    # Read inputs and propogate time-independent variables in time
    raw_inputs = pd.read_csv(input_path)
    inputs = np.repeat(raw_inputs.values[:, np.newaxis, :], outputs.shape[1], axis=1)

    # Combine into one data set
    all_data = np.concatenate((inputs, outputs), axis=-1)

    # If we are not stacking the time series then return 3D data else we
    # stack and return 2D data
    if not stack_series:
        nominal_data = xr.DataArray(
            all_data[[0],],
            coords={
                "samples": [0],
                "time steps": np.arange(all_data.shape[1]),
                "features": list(raw_inputs.columns) + list(output_paths.keys()),
            },
        )
        perturbed_data = xr.DataArray(
            all_data[1:,],
            coords={
                "samples": np.arange(all_data.shape[0] - 1),
                "time steps": np.arange(all_data.shape[1]),
                "features": list(raw_inputs.columns) + list(output_paths.keys()),
            },
        )
        return nominal_data, perturbed_data

    else:
        nominal_data = xr.DataArray(
            all_data[[0],].reshape((-1, all_data.shape[-1])),
            coords={
                "time steps": np.arange(all_data.shape[1]),
                "features": list(raw_inputs.columns) + list(output_paths.keys()),
            },
        )
        perturbed_data = xr.DataArray(
            all_data[1:,].reshape((-1, all_data.shape[-1])),
            coords={
                "time steps": np.arange((all_data.shape[0] - 1) * all_data.shape[1]),
                "features": list(raw_inputs.columns) + list(output_paths.keys()),
            },
        )
        return nominal_data, perturbed_data
