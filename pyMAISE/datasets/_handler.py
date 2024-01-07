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
def load_BWR():
    """
    Load BWR Micro Core data. This data consists of 2000 samples of 9 inputs:
    
    - ``PSZ``: Fuel bundle region Power Shaping Zone (PSZ),
    - ``DOM``:  Fuel bundle region Dominant zone (DOM),
    - ``vanA``: Fuel bundle region vanishing zone A (VANA),
    - ``vanB``: Fuel bundle region vanishing zone B (VANB),
    - ``subcool``: Represents moderator inlet conditions. Core inlet subcooling interpreted to be at the
      steam dome pressure (i.e., not core-averaged pressure). The input value for subcooling
      will automatically be increased to account for this fact. (Btu/lb),
    - ``CRD``: Defines the position of all control rod groups (banks),
    - ``flow_rate``: Defines essential global design data for rated coolant mass flux for the active core,
      :math:`\\frac{kg}{(cm^{2}-hr)}`. Coolant   mass flux equals active core flow divided by
      core cross-section area. Core cross-section area is DXA 2 times the number of assemblies,
    - ``power_density``: Defines essential global design data for rated power density using cold dimensions,
      :math:`(\\frac{kw}{liter})`,
    - ``VFNGAP``: Defines the ratio of narrow water gap width to the sum of the narrow and wide water
      gap widths,

    with 5 outputs:

    - ``K-eff``:  Reactivity coefficient k-effective, the effective neutron multiplication factor,
    - ``Max3Pin``: Maximum planar-averaged pin power peaking factor,
    - ``Max4Pin``: maximum pin-power peaking factor, :math:`F_{q}`, (which includes axial intranodal peaking),
    - ``F-delta-H``: Ratio of max-to-average enthalpy rise in a channel,
    - ``Max-Fxy``: Maximum radial pin-power peaking factor,

    This data set was constructed through uniform and normal sampling of the 9 input parameters for a boiling water
    reactor (BWR) micro-core. These samples were then used to solve for reactor characteristic changes in heat
    distribution and neutron flux. This BWR micro-core consists of 4 radially and axially heterogenous assemblies of
    the same type constructed in a 2x2 grid with a control blade placed in the center. A single assembly was brocken
    into seven zones where each zones 2D radial cross
    sectional information was constructed using CASMO-4. These cross sectional libraries were then processed through
    CMSLINK for SIMULATE-3 to interpret. The core geometry and physics was implemented and modeled using SIMULATE-3.
    
    Returns
    -------
    data: xarray.DataArray
        Raw BWR Micro Reactor data.
    inputs: xarray.DataArray
        9 inputs.
    outputs: xarray.DataArray
        5 outputs.
    """
    return read_csv(
        [_get_full_path("datasets/bwr_input.csv"), _get_full_path("datasets/bwr_output.csv")],
    )

# Load HTGR data
def load_HTGR():
    """
    Load HTGR Micro Reactor data. This data consists of 3000 samples of 8 inputs:
   
    - ``theta_{1}``: Angle of control drum in quadrant 1 (degrees), 
    - ``theta_{2}``: Angle of control drum in quadrant 1 (degrees), 
    - ``theta_{3}``: Angle of control drum in quadrant 2 (degrees),  
    - ``theta_{4}``: Angle of control drum in quadrant 2 (degrees),
    - ``theta_{5}``: Angle of control drum in quadrant 3 (degrees),
    - ``theta_{6}``: Angle of control drum in quadrant 3 (degrees),
    - ``theta_{7}``: Angle of control drum in quadrant 4 (degrees), 
    - ``theta_{8}``: Angle of control drum in quadrant 4 (degrees),  
    
    with 4 outputs:

    - ``FluxQ1``: Neutron flux in quadrant 1 :math:`(\\frac{neutrons}{cm^{2} s})`,
    - ``FluxQ2``: Neutron flux in quadrant 2 :math:`(\\frac{neutrons}{cm^{2} s})`,
    - ``FluxQ3``: Neutron flux in quadrant 3 :math:`(\\frac{neutrons}{cm^{2} s})`,
    - ``FluxQ4``: Neutron flux in quadrant 4 :math:`(\\frac{neutrons}{cm^{2} s})`,

    The data set featured in this work was based around the
    HOLOS-Quad reactor design. This reactor implements modular construction where seperate units can be transported
    independently and assembled at the specified location.  The HOLOS-Quad core is specifically a 22 MWt high-temperature
    gas-cooled microreactor (HTGR) which is controlled by 8 cylindrical control drums. It utilizes TRISO fuel particles
    contained in hexagonal graphite blocks used as a moderator. These graphite blocks have channels where helium gas can
    pass through for cooling.  The main importance of this data set is the influence on the control drums on the neutron
    flux. The drums control reactivity by rotating to vary the proximity of :math:`B_{4} C` on a portion of their outer edges to
    the fueled region of the core. Perturbations of the control drums in tern causes the core power shape to shift leading
    to complexe power distributions. Therefore, predictions of control drum reactivity worths for arbitrary configurations
    makes this problem nontrivial.
    
    Returns
    -------
    data: xarray.DataArray
        Raw HTGR Micro Reactor data with no symmetry conditions applied.
    inputs: xarray.DataArray
        8 inputs.
    outputs: xarray.DataArray
        4 outputs.
    """

    return  read_csv(_get_full_path("datasets/HTGR_microreactor.csv"), slice(29, 37), slice(4, 8))

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
