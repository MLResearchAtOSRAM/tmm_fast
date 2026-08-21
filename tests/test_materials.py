import numpy as np
import pytest

from tmm_fast import materials


@pytest.mark.parametrize('wavelength', [0.0, -1e-9, np.nan, np.inf, -np.inf])
def test_nonpositive_or_nonfinite_wavelength_is_rejected(wavelength):
    with pytest.raises(ValueError):
        materials.load('SiO2', [wavelength])
