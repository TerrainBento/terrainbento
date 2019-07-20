# coding: utf8
# !/usr/env/python
r""" **PrecipChanger** changes precipitation frequency and intensity over time.

This terrainbento boundary-condition handler was designed to change the
precipitation frequency and intensity over time in order to modify the water
erodibility coefficient.

In order to accomplish this, we need a theory by which to relate changes in
the precipitation to changes in erodibility.

We start by describing the assumed precipitation model.

The precipitation model considers the fraction of wet days, :math:`F`, and the
frequency distribution of precipitation depth on wet days. Precipitation on a
given day is given as :math:`p`. For daily average precipitation intensity, we
assume that the complementary cumulative distribution function is a stretched
exponential:

.. math::

    Pr(P>p) = \exp \left[-\left( p \lambda \right)^c\right]

where :math:`c` is the shape factor and :math:`\lambda` is the scale factor.
The corresponding probability density function is a Weibull distribution. The
mean wet-day precipitation depth :math:`p_d` is related to the scale factor by

.. math::

    p_d = \lambda \Gamma (1 + 1/c)

where :math:`\Gamma` is the gamma function.

The drainage area-based basic erosion law considered here is:

.. math::

    E = KA^{m}S^{n}

where :math:`E` is channel erosion rate, :math:`A` is contributing drainage
area, and :math:`S` is local channel gradient. :math:`m` and :math:`n` are the
slope and area exponents

With :math:`m=1`, :math:`K` has dimensions of inverse length.

Here, we present the approach used to relate changes in :math:`K` to changes in
:math:`p_d`

Deriving a relation between :math:`K`, :math:`p_d`, and :math:`F` requires
defining an underlying hydrology model. We start by noting that drainage area
serves as a surrogate for discharge, :math:`Q`. We can therefore write an
`instantaneous` version of the erosion law as:

.. math::

    E_i = K_q Q^{m}S^n.

This formulation represents the erosion rate during a particular daily event,
:math:`E_i` with daily-average discharge :math:`Q_q`, as opposed to the
long-term average rate of erosion, :math:`E`. It introduces a new term
:math:`K_q`, the daily-averaged erosion coefficent.

We next assume that discharge is the product of runoff rate, :math:`r`, and
drainage area:

.. math::

    Q = r A.

Combining these we can write

.. math::

    E_i = K_q r^{m} Q^{m} S^{m}.

This equation establishes the dependence of short-term erosion rate on
catchment-average runoff rate, :math:`r`.

Next we need to relate runoff rate to precipitation rate. A common method is to
acknowledge that there exists a soil infiltration capacity, :math:`I_c`, such
that when :math:`p<I_c`, no runoff occurs, and when :math:`p>I_c`,

.. math::

    r = p - I_c.

An advantage of this simple approach is that :math:`I_c` can be measured
directly or inferred from stream-flow records.

To relate short-term ("instantaneous") erosion rate to the long-term average,
one can first integrate the erosion rate over the full probability distribution
of daily precipitation intensity. This operation yields the average erosion
rate produced on wet days. To convert this into an average that includes dry
days, we simply multiply the integral by the wet-day fraction :math:`F`. Thus,
the long-term erosion rate by water can be expressed as:

.. math::

    E = F \int_{I_c}^\infty K_q (p-I_c)^{m}Q^{m} S^{n} f(p) dp,

where :math:`f(p)` is the probability density function (PDF) of daily
precipitation intensity. By equating the above definition of long-term erosion
math:`E` with the simpler definition :math:`E = K Q^{m}S^{n}`,
we can solve for the effective erosion coefficient, :math:`K`:

.. math::

    K = F K_q \int_{I_c}^\infty (p-I_c)^{m} f(p) dp.

In this case, what is of interest is the `change` in :math:`K` given some
change in precipitation frequency distribution :math:`f(p)`. Suppose we have an
original value of the effective erodibility coefficient, :math:`K_0`, and an
original precipitation distribution, :math:`f_0(p)`. Given a future change to a
new precipitation distribution :math:`f(p)`, we wish to know what is the ratio
of the new effective erodibility coefficient :math:`K` to its original value.
Using the definition of :math:`K` above, the ratio of old to new coefficient
is:

.. math::

    \frac{K}{K_0} =
    \frac{F\int_{I_c}^\infty (p-I_c)^{m} f(p) dp}
          {F_0\int_{I_c}^\infty (p-I_c)^{m} f_0(p) dp}

Here :math:`F_0` is the starting intermittency factor.

Thus, if we know the original and new precipitation distributions and
intermittency factors, we can determine the resulting change in :math:`K`.

We assume that the daily precipitation intensity PDF is given by the Weibull
distribution such that :math:`f(p)` has the form:

.. math::

    f(p) = \frac{c}
                 {\lambda}\left( \frac{p}{\lambda} \right)^{(c-1)}
                 e^{-(p \lambda)^c}.

The above definition can be substituted in the integrals in the equation for
:math:`\frac{K}{K_0}`. We are not aware of a closed-form solution to the
resulting integrals. Therefore, we apply a numerical integration to convert the
input values of :math:`F`, :math:`c`, and :math:`p_d` into a corresponding new
value of :math:`K`.

For computational convenience, we define and calculate :math:`\Psi` which
represents the portion of the erosion coefficient that depends on
precipitation.

:math:`\Psi` is defined as the integral from :math:`I_c` to infinity of the
rainfall in excess of infiltration.

.. math::

    \Psi = \int_{I_c}^\infty (p - I_{c})^m f(p) dp

Finally we define the erodibility adjustment factor :math:`F_{w}`:

.. math::

     K = F_{w} K_{0} = \frac{F \Psi}{F_0 \Psi_0} K_{0}

Here :math:`F_0` and :math:`\Psi_0` are the starting fraction of wet days and
starting value for :math:`\Psi`.

**PrecipChanger** presently supports changes in :math:`F` and :math:`p_d` but
not :math:`c`.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def _integrand(p, Ic, lam, c, m):
    """Calculate the integrand for numerical integration.

    Calculates the value of

    .. math::

        (p-I_{c})^{m} f(p)

    where :math:`f(p)`` is a Weibull distribution.

    Called by the scipy "quad" numerical integration function.

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
    return (
        ((p - Ic) ** m)
        * (c / lam)
        * ((p / lam) ** (c - 1.0))
        * np.exp(-((p / lam) ** c))
    )


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


def _check_intermittency_value(rainfall_intermittency_factor):
    """Check that rainfall_intermittency_factor is >= 0 and <=1."""
    if (rainfall_intermittency_factor < 0.0) or (
        rainfall_intermittency_factor > 1.0
    ):
        raise ValueError(
            (
                "The PrecipChanger rainfall_intermittency_factor has a "
                "value of less than zero or greater than one. "
                "This is invalid."
            )
        )


def _check_mean_depth(mean_depth):
    """Check that mean depth is >= 0."""
    if mean_depth < 0:
        raise ValueError(
            (
                "The PrecipChanger mean depth has a "
                "value of less than zero. This is invalid."
            )
        )


def _check_infiltration_capacity(infiltration_capacity):
    """Check that infiltration_capacity >= 0."""
    if infiltration_capacity < 0:
        raise ValueError(
            (
                "The PrecipChanger infiltration_capacity has a "
                "value of less than zero. This is invalid."
            )
        )


class PrecipChanger(object):
    """Handle time varying precipitation.

    The **PrecipChanger** handles time-varying precipitation by changing the
    proportion of time rain occurs
    (``daily_rainfall_rainfall_intermittency_factor``)
    and the mean of the daily rainfall Weibull distribution
    (``rainfall__mean_rate``).

    Note that **PrecipChanger** increments time at the end of the
    **run_one_step** method.
    """

    def __init__(
        self,
        grid,
        daily_rainfall__intermittency_factor=None,
        daily_rainfall__intermittency_factor_time_rate_of_change=None,
        rainfall__mean_rate=None,
        rainfall__mean_rate_time_rate_of_change=None,
        rainfall__shape_factor=None,
        infiltration_capacity=None,
        m_sp=0.5,
        precipchanger_start_time=0,
        precipchanger_stop_time=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        grid : landlab model grid
        daily_rainfall_intermittency_factor : float
            Starting value of the daily rainfall intermittency factor
            :math:`F`. This value is a proportion and ranges from 0 (no rain
            ever) to 1 (rains every day).
        daily_rainfall_intermittency_factor__time_rate_of_change : float
            Time rate of change of the daily rainfall intermittency factor
            :math:`F`. Units are implied by the ``time_unit`` argument. Note
            that this factor must always be between 0 and 1.
        rainfall__mean_rate : float
            Starting value of the mean daily rainfall intensity :math:`p_d`.
            Units are implied by the ``time_unit`` argument.
        rainfall__mean_rate__time_rate_of_change : float
            Time rate of change of the mean daily rainfall intensity
            :math:`p_d`. Units are implied by the ``time_unit`` argument.
        rainfall__shape_factor : float
            Weibull distribution shape factor :math:`c`.
        infiltration_capacity : float
            Infiltration capacity. Time units are implied by the ``time_unit``
            argument.
        m_sp : float, optional
            Drainage area exponent in erosion rule, :math:`m`.  Default value
            is 0.5.
        precipchanger_start_time : float, optional
            Model time at which changing the precipitation should start.
            Default is at the onset of the model run.
        precipchanger_stop_time : float, optional
            Model time at which changing the precipitation statistics should
            end. Default is no end time.

        Notes
        -----
        The time units of ``rainfall__mean_rate``,
        ``rainfall__mean_rate_time_rate_of_change``, and
        ``infiltration_capacity`` are all assumed to be the same.

        The value passed by ``time_unit`` is assumed to be consistent with
        the time units of `step`.

        The length units are assumed to be consistent with the model grid
        definition.

        Examples
        --------
        Start by creating a landlab model grid.

        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((5, 5))

        Now import the **PrecipChanger** and instantiate.

        >>> from terrainbento.boundary_handlers import PrecipChanger
        >>> bh = PrecipChanger(mg,
        ...    daily_rainfall__intermittency_factor = 0.3,
        ...    daily_rainfall__intermittency_factor_time_rate_of_change = 0.01,
        ...    rainfall__mean_rate = 3.0,
        ...    rainfall__mean_rate_time_rate_of_change = 0.2,
        ...    rainfall__shape_factor = 0.65,
        ...    infiltration_capacity = 0)

        We can get the current precipitation parameters

        >>> I, pd = bh.get_current_precip_params()
        >>> print(I)
        0.3

        Note that ``rainfall__mean_rate`` is provided in units of
        length per year.

        >>> print(pd)
        3.0

        Since we did not specify a start time or stop time the PrecipChanger
        will immediate start to modify the values of precipitation parameters.

        >>> bh.run_one_step(10.0)
        >>> I, pd = bh.get_current_precip_params()
        >>> print(I)
        0.4
        >>> print(pd)
        5.0

        If we are using an erosion model that requires the raw values of the
        precipitation parameters, we can use them. If instead we are using
        a model that does not explicitly treat event-scale precipitation, we
        can use the bulk erodibility adjustment factor :math:`F_w`.

        >>> fw = bh.get_erodibility_adjustment_factor()
        >>> print(round(fw, 3))
        1.721
        """
        if daily_rainfall__intermittency_factor is None:
            msg = (
                "terrainbento PrecipChanger requires the parameter "
                "daily_rainfall__intermittency_factor"
            )
            raise ValueError(msg)

        if daily_rainfall__intermittency_factor_time_rate_of_change is None:
            msg = (
                "terrainbento PrecipChanger requires the parameter "
                "daily_rainfall__intermittency_factor_time_rate_of_change"
            )
            raise ValueError(msg)

        if rainfall__mean_rate is None:
            msg = (
                "terrainbento PrecipChanger requires the parameter "
                "rainfall__mean_rate"
            )
            raise ValueError(msg)

        if rainfall__mean_rate_time_rate_of_change is None:
            msg = (
                "terrainbento PrecipChanger requires the parameter "
                "rainfall__mean_rate_time_rate_of_change"
            )
            raise ValueError(msg)

        if rainfall__shape_factor is None:
            msg = (
                "terrainbento PrecipChanger requires the parameter "
                "rainfall__shape_factor"
            )
            raise ValueError(msg)

        if infiltration_capacity is None:
            msg = (
                "terrainbento PrecipChanger requires the parameter "
                "infiltration_capacity"
            )
            raise ValueError(msg)

        self.model_time = 0.0

        if precipchanger_stop_time is None:
            self.no_stop_time = True
        else:
            self.no_stop_time = False
            self.stop_time = precipchanger_stop_time
        self.start_time = precipchanger_start_time

        self.starting_frac_wet_days = daily_rainfall__intermittency_factor
        self.frac_wet_days_rate_of_change = (
            daily_rainfall__intermittency_factor_time_rate_of_change
        )

        self.starting_daily_mean_depth = rainfall__mean_rate
        self.mean_depth_rate_of_change = (
            rainfall__mean_rate_time_rate_of_change
        )

        self.rainfall__shape_factor = rainfall__shape_factor
        self.infilt_cap = infiltration_capacity
        self.m = m_sp

        self.starting_psi = self.calculate_starting_psi()

        _check_intermittency_value(self.starting_frac_wet_days)
        _check_mean_depth(self.starting_daily_mean_depth)
        _check_infiltration_capacity(self.infilt_cap)

    def calculate_starting_psi(self):
        r"""Calculate and store for later the factor :math:`\Psi_0`.

        :math:`\Psi` represents the portion of the erosion coefficient that
        depends on precipitation intensity. :math:`\Psi_0` is the starting
        value of :math:`\Psi`.

        :math:`\Psi_0` is defined as the integral from :math:`I_c` to infinity
        of the rainfall in excess of infiltration.

        .. math::

            \Psi_0 = \int_{I_c}^\infty (p - I_{c})^m f_0(p) dp

        where :math:`p` is precipitation intensity, :math:`I_c` is infiltration
        capacity, :math:`m` is the discharge/area exponent (e.g., 1/2), and
        :math:`f_0(p)` is the Weibull distribution representing the probability
        distribution of daily precipitation intensity at model run onset.
        """
        lam = _scale_fac(
            self.starting_daily_mean_depth, self.rainfall__shape_factor
        )
        psi, _ = quad(
            _integrand,
            self.infilt_cap,
            np.inf,
            args=(self.infilt_cap, lam, self.rainfall__shape_factor, self.m),
        )
        return psi

    def get_current_precip_params(self):
        """Return current values precipitation parameters.

        Returns
        -------
        daily_rainfall_rainfall_intermittency_factor : float
        rainfall__mean_rate : float
        """
        # if after start time
        if self.model_time > self.start_time:

            # get current evaluation time
            if self.no_stop_time:
                time = self.model_time
            else:
                if self.model_time > self.stop_time:
                    time = self.stop_time
                else:
                    time = self.model_time

            # calculate and return updated values
            frac_wet_days = (
                self.starting_frac_wet_days
                + self.frac_wet_days_rate_of_change * time
            )
            mean_depth = (
                self.starting_daily_mean_depth
                + self.mean_depth_rate_of_change * time
            )

            _check_intermittency_value(frac_wet_days)
            _check_mean_depth(mean_depth)

            return frac_wet_days, mean_depth
        else:
            # otherwise return starting values.
            return self.starting_frac_wet_days, self.starting_daily_mean_depth

    def get_erodibility_adjustment_factor(self):
        r"""Calculates the erodibility adjustment factor at the current time.

        Calculates and returns the factor :math:`F_{w}` by which an erodibility
        by water should be adjusted.

        .. math::

             K = F_{w} K_{0} = \frac{F \Psi}{F_0 \Psi_0} K_{0}

        Returns
        -------
        erodibility_adjustment_factor : float
        """
        # if after start time
        if self.model_time > self.start_time:

            # get the updated precipitation parameters
            frac_wet, mean_depth = self.get_current_precip_params()

            # calculate the mean intensity and the scale factor
            lam = _scale_fac(mean_depth, self.rainfall__shape_factor)

            # calculate current value of Psi
            psi, _ = quad(
                _integrand,
                self.infilt_cap,
                np.inf,
                args=(
                    self.infilt_cap,
                    lam,
                    self.rainfall__shape_factor,
                    self.m,
                ),
            )

            # calculate the adjustment factor
            adj_fac = (frac_wet * psi) / (
                self.starting_frac_wet_days * self.starting_psi
            )
            # and return
            return adj_fac
        else:
            # if before starting time, return 1.0
            return 1.0

    def run_one_step(self, step):
        """Run **PrecipChanger** forward and update model time.

        The **run_one_step** method provides a consistent interface to update
        the terrainbento boundary condition handlers.

        In the **run_one_step** routine, the **PrecipChanger** will update its
        internal record of model time.

        Parameters
        ----------
        step : float
            Duration of model time to advance forward.
        """
        self.model_time += step
