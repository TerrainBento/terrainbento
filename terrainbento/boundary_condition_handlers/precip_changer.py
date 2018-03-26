#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
PrecipChanger controls precipitation frequency and/or intensity over time.
"""

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad


DAYS_PER_YEAR = 365.25


def _integrand(p, Ic, lam, c, m):
    """Calculates the value of

    .. math::
        (p-I_{c})^{m} f(p)

    where .. math::`f(p)`` is a Weibull distribution.

    Called by the scipy 'quad' numerical integration function.

    Parameters
    ----------
    p : float
        Precipitation intensity
    Ic : float
        Infiltration capacity
    lam : float
        Weibull distribution scale factor
    c : float
        Weibull distribution shape factor
    m : float
        Drainage area exponent

    Examples
    --------
    from numpy.testing import assert_almost_equal
    integrand = _integrand(5.0, 1.0, 0.5, 0.5, 0.5)
    assert_almost_equal(integrand, 0.026771349117364424)
    """
    return (((p - Ic) ** m) * (c / lam) * ((p / lam) ** (c - 1.0))
            * np.exp(-((p / lam) ** c)))

def _scale_fac(pmean, c):
    """Convert mean precipitation intensity into Weibull scale factor lambda.

    Parameters
    ----------
    pmean : float
        Mean precipitation intensity
    c : float
        Weibull distribution shape factor

    Examples
    --------
    from numpy.testing import assert_almost_equal
    scale_factor = _scale_fac(1.0, 0.6)
    assert_almost_equal(scale_factor, 0.66463930045948338)
    """
    return pmean * (1.0 / gamma(1.0 + 1.0 / c))


def _depth_to_intensity(depth, time_unit):
    """Convert daily precip water depth to intensity using given time units.

    Parameters
    --------
    depth : float
        Daily precipitation intensity
    time_unit : str
        Unit of time. Presently only 'year' is supported

    Examples
    --------
    from numpy.testing import assert_almost_equal
    intensity = _depth_to_intensity(2.0, 'year')
    assert_almost_equal(intensity, 2*365.25)
    """
    if time_unit == 'year':
        intensity = depth * DAYS_PER_YEAR
    else:
        raise NotImplementedError(('The PrecipChanger presently only supports '
                                    'time units of year'))

    return intensity


class PrecipChanger(object):
    """Class PrecipChanger handles time-varying precipitation and related
    parameters in Erosion Modeling Suite (EMS).
    """

    def __init__(self,
                 grid,
                 intermittency_factor = 0.3,
                 intermittency_factor_rate_of_change = 0.001,
                 mean_storm__intensity = 0.3,
                 mean_depth_rate_of_change = 0.001,
                 precip_shape_factor = 0.65,
                 time_unit = 'year',
                 infiltration_capacity = None,
                 m_sp = None,
                 precip_stop_time = None,
                 length_factor = 1.0
                 **params):

        """Initialize a PrecipChanger object.
        """
        self.params = params

        if precip_stop_time is None:
            stop_time = kwargs['run_duration']

        self.starting_frac_wet_days = intermittency_factor
        self.frac_wet_days_rate_of_change = intermittency_factor_rate_of_change

        self.starting_daily_mean_depth = mean_storm__intensity / DAYS_PER_YEAR * self._length_factor
        self.mean_depth_rate_of_change = mean_depth_rate_of_change / DAYS_PER_YEAR * self._length_factor
        self.precip_shape_factor = precip_shape_factor
        self.time_unit = time_unit
        try:
            self.infilt_cap = infiltration_capacity * self._length_factor
        except TypeError:
            self.infilt_cap = infiltration_capacity
        self.m = m_sp
        self.stop_time = stop_time

        if self.infilt_cap is not None:
            (self.starting_psi, abserr) = self._calculate_starting_psi()

    def _calculate_starting_psi(self):
        """Calculate and store for later the factor psi, which represents the
        portion of the erosion coefficient that depends on precipitation
        intensity.

        Psi is defined as the integral from Ic to infinity of

            (p - Ic)^m f(p) dp

        where p is precipitation intensity, Ic is infiltration capacity, m is
        the discharge/area exponent (e.g., 1/2), and f(p) is the Weibull
        distribution representing the probability distribution of daily
        precipitation intensity.
        """
        mean_intensity = _depth_to_intensity(self.starting_daily_mean_depth,
                                            self.time_unit)
        lam = _scale_fac(mean_intensity, self.precip_shape_factor)
        psi = quad(_integrand, self.infilt_cap, np.inf,
                         args=(self.infilt_cap, lam, self.precip_shape_factor,
                               self.m))
        return psi

    def get_current_precip_params(self, current_time):
        """Return current frac wet days and daily mean depth."""

        if current_time > self.stop_time:
            current_time = self.stop_time

        frac_wet_days = (self.starting_frac_wet_days
                         + self.frac_wet_days_rate_of_change * current_time)
        mean_depth = (self.starting_daily_mean_depth
                         + self.mean_depth_rate_of_change * current_time)
        return frac_wet_days, mean_depth

    def get_erodibility_adjustment_factor(self, current_time):
        """Calculates and returns the factor by which erodibility ("K")
        should be adjusted.

        Erodibility factor K is defined here as in the docstring above:

             K = Fw Kq psi

        We will have already calculated Kq, and it won't
        """

        if current_time > self.stop_time:
            current_time = self.stop_time

        frac_wet, mean_depth = self.get_current_precip_params(current_time)

        mean_intensity = _depth_to_intensity(mean_depth, self.time_unit)
        lam = _scale_fac(mean_intensity, self.precip_shape_factor)
        (psi, abserr) = quad(_integrand, self.infilt_cap, np.inf,
                         args=(self.infilt_cap, lam, self.precip_shape_factor,
                               self.m))
        adj_fac = ((frac_wet * psi)
                    / (self.starting_frac_wet_days * self.starting_psi))
        return adj_fac

    def run_one_step(self, dt):
        """ """
        pass
