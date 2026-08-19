"""
Refractive index data for a few materials commonly used in thin-film stacks.

The data files next to this module hold three columns: wavelength in nanometer,
n and k of the refractive index n + 1j*k.
"""

from importlib.resources import files
from warnings import warn

import numpy as np

__all__ = ['available', 'load']

_PREFIX = 'n'
_SUFFIX = '.txt'
_IMPLAUSIBLE_WAVELENGTH = 1e-3


def available():
    """Names of the bundled materials, as accepted by load()."""
    return sorted(entry.name[len(_PREFIX):-len(_SUFFIX)]
                  for entry in files(__name__).iterdir()
                  if entry.name.startswith(_PREFIX) and entry.name.endswith(_SUFFIX))


def _datafile(name):
    """Data file of the material name, looked up case insensitively."""
    wanted = (_PREFIX + str(name) + _SUFFIX).lower()
    for entry in files(__name__).iterdir():
        if entry.name.lower() == wanted:
            return entry
    raise ValueError("unknown material '" + str(name) + "', available materials are: "
                     + ', '.join(available()))


def load(name, wl, complex_n=True):
    """
    Refractive index of a bundled material, interpolated onto given wavelengths.

    Parameters:
    -----------
    name : str
        Material as listed by available(), matched case insensitively, e.g. 'SiO2'
    wl : array_like
        Vacuum wavelengths in meter of shape [W]
    complex_n : bool
        Whether to return n + 1j*k or only the real part n

    Returns:
    --------
    np.ndarray of shape [W]
        Refractive index at each requested wavelength

    Raises:
    -------
    ValueError
        If the material is unknown, or if wl was not given in meter
    """
    wl = np.asarray(wl, dtype=float).reshape(-1)
    if wl.size == 0:
        raise ValueError('wl must hold at least one wavelength')
    if np.nanmax(wl) > _IMPLAUSIBLE_WAVELENGTH:
        raise ValueError('wl must be given in meter but holds values up to '
                         + str(np.nanmax(wl)) + '; for wavelengths in nanometer pass wl * 1e-9')
    with _datafile(name).open('r') as datafile:
        table = np.loadtxt(datafile)
    table = table[np.argsort(table[:, 0])]  # np.interp requires ascending wavelengths
    tabulated_wl = table[:, 0]

    wl_nm = wl * 1e9
    if wl_nm.min() < tabulated_wl[0] or wl_nm.max() > tabulated_wl[-1]:
        warn('Wavelengths from ' + str(round(float(wl_nm.min()), 1)) + ' to '
             + str(round(float(wl_nm.max()), 1)) + ' nm were requested, but ' + str(name)
             + ' is only tabulated from ' + str(tabulated_wl[0]) + ' to ' + str(tabulated_wl[-1])
             + ' nm; the outermost tabulated values are used beyond that range.')

    n = np.interp(wl_nm, tabulated_wl, table[:, 1])
    if not complex_n:
        return n
    return n + 1j * np.interp(wl_nm, tabulated_wl, table[:, 2])
