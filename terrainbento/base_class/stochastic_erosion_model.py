# coding: utf8
# !/usr/env/python
"""Base class for common functions of terrainbento stochastic erosion
models."""

import os
import textwrap

import numpy as np
import scipy.stats as stats

from landlab.components import PrecipitationDistribution
from terrainbento.base_class import ErosionModel

_STRING_LENGTH = 80


class StochasticErosionModel(ErosionModel):
    """Base class for stochastic-precipitation terrainbento models.

    A **StochasticErosionModel** inherits from
    :py:class:`ErosionModel` and provides functionality needed by all
    stochastic-precipitation models.

    This is a base class that handles processes related to the generation of
    preciptiation events.

    Two primary options are avaliable for the stochastic erosion models. When
    ``opt_stochastic_duration=True`` the model will use the
    `PrecipitationDistribution <https://landlab.readthedocs.io/en/latest/landlab.components.uniform_precip.html>`_
    Landlab component to generate a random storm duration, interstorm duration,
    and precipitation intensity or storm depth from an exponential
    distribution. When this option is selected, the following parameters are
    used:

        - mean_storm_duration
        - mean_interstorm_duration
        - mean_storm_depth

    When ``opt_stochastic_duration==False`` the model will have uniform
    timesteps but generate rainfall from a stretched exponential distribution.
    The duration indicated by the parameter ``step`` will first be split into a
    series of sub-timesteps based on the parameter
    ``number_of_sub_time_steps``, and then each of these sub-timesteps will
    experience a duration of rain and no-rain based on the value of
    ``rainfall_intermittency_factor``. The duration of rain and no-rain will
    not change, but the intensity of rain will vary based on a stretched
    exponential distribution described by the shape factor
    ``rainfall__shape_factor`` and with a scale factor calculated so that the
    mean of the distribution has the value given by ``rainfall__mean_rate``.

    The following parameters are used:

        - rainfall__shape_factor
        - number_of_sub_time_steps
        - rainfall_intermittency_factor
        - rainfall__mean_rate

    The hydrology uses calculation of drainage area using the user-specified
    routing method. It then performs one of two options, depending on the
    user's choice of ``opt_stochastic_duration`` (True or False).

    If the user requests stochastic duration, the model iterates through a sequence
    of storm and interstorm periods. Storm depth is drawn at random from a gamma
    distribution, and storm duration from an exponential distribution; storm
    intensity is then depth divided by duration. This sequencing is implemented by
    overriding the run_for method.

    If the user does not request stochastic duration (indicated by setting
    ``opt_stochastic_duration`` to ``False``), then the default
    (**erosion_model** base class) **run_for** method is used. Whenever
    **run_one_step** is called, storm intensity is generated at random from an
    exponential distribution with mean given by the parameter
    ``rainfall__mean_rate``. The stream power component is run for only a
    fraction of the time step duration step, as specified by the parameter
    ``rainfall_intermittency_factor``. For example, if ``step`` is 10 years and
    the intermittency factor is 0.25, then the stream power component is run
    for only 2.5 years.

    In either case, given a storm precipitation intensity :math:`P`, the runoff
    production rate :math:`R` [L/T] is calculated using:

    .. math::

        R = P - I (1 - \exp ( -P / I ))

    where :math:`I` is the soil infiltration capacity. At the sub-grid scale, soil
    infiltration capacity is assumed to have an exponential distribution of which
    :math:`I` is the mean. Hence, there are always some spots within any given grid cell
    that will generate runoff. This approach yields a smooth transition from
    near-zero runoff (when :math:`I>>P`) to :math:`R \\approx P`
    (when :math:`P>>I`), without a "hard threshold."

    The following at-node fields must be specified in the grid:
        - ``topographic__elevation``
    """

    _required_fields = ["topographic__elevation"]

    def __init__(
        self,
        clock,
        grid,
        random_seed=0,
        record_rain=False,
        opt_stochastic_duration=False,
        mean_storm_duration=1,
        mean_interstorm_duration=1,
        mean_storm_depth=1,
        rainfall__shape_factor=1,
        number_of_sub_time_steps=1,
        rainfall_intermittency_factor=1,
        rainfall__mean_rate=1,
        storm_sequence_filename="storm_sequence.txt",
        frequency_filename="exceedance_summary.txt",
        **kwargs
    ):
        """
        Parameters
        ----------
        clock : terrainbento Clock instance
        grid : landlab model grid instance
            The grid must have all required fields.
        random_seed, int, optional
            Random seed. Default is 0.
        opt_stochastic_duration : bool, optional
            Flag indicating if timestep is stochastic or constant. Default is
            False.
        mean_storm_duration : float, optional
            Average duration of a precipitation event. Default is 1.
        mean_interstorm_duration : float, optional
            Average duration between precipitation events. Default is 1.
        mean_storm_depth : float, optional
            Average depth of precipitation events. Default is 1.
        number_of_sub_time_steps : int, optional
            Number of sub-timesteps. Default is 1.
        rainfall_intermittency_factor : float, optional
            Value between zero and one that indicates the proportion of time
            rain occurs. A value of 0 means it never rains and a value of 1
            means that rain never ceases.  Default is 1.
        rainfall__mean_rate : float, optional
            Mean of the precipitation distribution.  Default is 1.
        rainfall__shape_factor : float, optional
            Shape factor of the precipitation distribution.  Default is 1.
        record_rain : boolean
            Flag to indicate if a sequence of storms should be saved. Default
            is False.
        storm_sequence_filename : str
            Storm sequence filename. Default is "storm_sequence.txt"
        frequency_filename : str
            Filename for precipitation exceedance frequency summary. Default
            value is "exceedance_summary.txt"
        **kwargs :
            Keyword arguments to pass to
            :py:class:`ErosionModel`

        Returns
        -------
        StochasticErosionModel : object

        Examples
        --------
        This model is a base class and is not designed to be run on its own. We
        recommend that you look at the terrainbento tutorials for examples of
        usage.
        """
        # Call StochasticErosionModel init
        super(StochasticErosionModel, self).__init__(clock, grid, **kwargs)

        # ensure Precipitator and RunoffGenerator are vanilla
        self._ensure_precip_runoff_are_vanilla()

        self.opt_stochastic_duration = opt_stochastic_duration

        # verify that opt_stochastic_duration and PrecipChanger are consistent
        if self.opt_stochastic_duration and (
            "PrecipChanger" in self.boundary_handlers
        ):
            msg = (
                "terrainbento StochasticErosionModel: setting "
                "opt_stochastic_duration=True and using the PrecipChanger "
                "boundary condition handler are not compatible."
            )
            raise ValueError(msg)

        self.seed = int(random_seed)

        self.random_seed = random_seed

        self.frequency_filename = frequency_filename
        self.storm_sequence_filename = storm_sequence_filename

        self.mean_storm_duration = mean_storm_duration
        self.mean_interstorm_duration = mean_interstorm_duration
        self.mean_storm_depth = mean_storm_depth
        self.shape_factor = rainfall__shape_factor
        self.number_of_sub_time_steps = number_of_sub_time_steps
        self.rainfall_intermittency_factor = rainfall_intermittency_factor
        self.rainfall__mean_rate = rainfall__mean_rate

        # initialize record for storms. Depending on how this model is run
        # (stochastic time, number_time_steps>1, more manually) the step may
        # change. Thus, rather than writing routines to reconstruct the time
        # series of precipitation from the step could change based on users use,
        # we"ll record this with the model run instead of re-running.

        # make this the non-default option.

        # Second, test that
        if record_rain:
            self.record_rain = True
            self.rain_record = {
                "event_start_time": [],
                "event_duration": [],
                "rainfall_rate": [],
                "runoff_rate": [],
            }
        else:
            self.record_rain = False
            self.rain_record = None

    def calc_runoff_and_discharge(self):
        """Calculate runoff rate and discharge; return runoff."""
        if self.rain_rate > 0.0 and self.infilt > 0.0:
            runoff = self.rain_rate - (
                self.infilt * (1.0 - np.exp(-self.rain_rate / self.infilt))
            )
            if runoff <= 0:
                runoff = 0  # pragma: no cover
        else:
            runoff = self.rain_rate
        self.grid.at_node["surface_water__discharge"][:] = (
            runoff * self.grid.at_node["drainage_area"]
        )
        return runoff

    def run_for_stochastic(self, step, runtime):
        """**Run_for** with stochastic duration.

        Run model without interruption for a specified time period, using
        random storm/interstorm sequence.

        **run_for_stochastic** runs the model for the duration ``runtime`` with
        model time steps given by the PrecipitationDistribution component.
        Model run steps will not exceed the duration given by ``step``.

        Parameters
        ----------
        step : float
            Model run timestep,
        runtime : float
            Total duration for which to run model.
        """
        self.rain_generator.delta_t = step
        self.rain_generator.run_time = runtime
        for (
            tr,
            p,
        ) in self.rain_generator.yield_storm_interstorm_duration_intensity():
            self.rain_rate = p
            self.run_one_step(tr)

    def instantiate_rain_generator(self):
        """Instantiate component used to generate storm sequence."""
        # Handle option for duration.
        if self.opt_stochastic_duration:
            self.rain_generator = PrecipitationDistribution(
                mean_storm_duration=self.mean_storm_duration,
                mean_interstorm_duration=self.mean_interstorm_duration,
                mean_storm_depth=self.mean_storm_depth,
                total_t=self.clock.stop,
                delta_t=self.clock.step,
                random_seed=self.seed,
            )
            self.run_for = self.run_for_stochastic  # override base method
        else:
            from scipy.special import gamma

            self.rain_generator = PrecipitationDistribution(
                mean_storm_duration=1.0,
                mean_interstorm_duration=1.0,
                mean_storm_depth=1.0,
                random_seed=self.seed,
            )

            self.scale_factor = self.rainfall__mean_rate / gamma(
                1.0 + (1.0 / self.shape_factor)
            )

            if (
                isinstance(self.number_of_sub_time_steps, (int, np.integer))
                is False
            ):
                raise ValueError(
                    ("number_of_sub_time_steps must be of type integer.")
                )

            self.n_sub_steps = self.number_of_sub_time_steps

    def reset_random_seed(self):
        """Reset the random number generation sequence."""
        self.rain_generator.seed_generator(seedval=self.seed)

    def _pre_water_erosion_steps(self):
        """Convenience function for pre-water erosion steps.

        If a model needs to do anything before each erosion step is run,
        e.g. recalculate a threshold value, that model should overwrite
        this function.
        """
        pass

    def handle_water_erosion(self, step, flooded):
        """Handle water erosion for stochastic models.

        If we are running stochastic duration, then self.rain_rate will
        have been calculated already. It might be zero, in which case we
        are between storms, so we don't do water erosion.

        If we're NOT doing stochastic duration, then we'll run water
        erosion for one or more sub-time steps, each with its own
        randomly drawn precipitation intensity.

        This routine assumes that a model-specific method:

                    **calc_runoff_and_discharge()**

        will have been defined. Additionally a model eroder must also have been
        defined.

        For example, BasicStVs calculated runoff and discharge in a different
        way than the other models.

        If the model has a function **update_threshold_field**, this
        function will test for it and run it. This is presently done in
        BasicDdSt.

        Parameters
        ----------
        step : float
            Model run timestep.
        flooded_nodes : ndarray of int (optional)
            IDs of nodes that are flooded and should have no erosion.
        """
        # (if we're varying precipitation parameters through time, update them)
        if "PrecipChanger" in self.boundary_handlers:
            self.rainfall_intermittency_factor, self.rainfall__mean_rate = self.boundary_handlers[
                "PrecipChanger"
            ].get_current_precip_params()

        if self.opt_stochastic_duration and self.rain_rate > 0.0:

            self._pre_water_erosion_steps()

            runoff = self.calc_runoff_and_discharge()

            self.eroder.run_one_step(step, flooded_nodes=flooded)
            if self.record_rain:
                # save record into the rain record
                self.record_rain_event(
                    self.model_time, step, self.rain_rate, runoff
                )

        elif self.opt_stochastic_duration and self.rain_rate <= 0.0:
            # calculate and record the time with no rain:
            if self.record_rain:
                self.record_rain_event(self.model_time, step, 0, 0)

        elif not self.opt_stochastic_duration:

            dt_water = (step * self.rainfall_intermittency_factor) / float(
                self.n_sub_steps
            )
            for i in range(self.n_sub_steps):
                self.rain_rate = self.rain_generator.generate_from_stretched_exponential(
                    self.scale_factor, self.shape_factor
                )

                self._pre_water_erosion_steps()

                runoff = self.calc_runoff_and_discharge()
                self.eroder.run_one_step(dt_water, flooded_nodes=flooded)
                # save record into the rain record
                if self.record_rain:
                    event_start_time = self.model_time + (i * dt_water)
                    self.record_rain_event(
                        event_start_time, dt_water, self.rain_rate, runoff
                    )

            # once all the rain time_steps are complete,
            # calculate and record the time with no rain:
            if self.record_rain:

                # calculate dry time
                dt_dry = step * (1 - self.rainfall_intermittency_factor)

                # if dry time is greater than zero, record.
                if dt_dry > 0:
                    event_start_time = self.model_time + (
                        self.n_sub_steps * dt_water
                    )
                    self.record_rain_event(event_start_time, dt_dry, 0.0, 0.0)

    def finalize(self):
        """Finalize stochastic erosion models.

        The finalization step of stochastic erosion models in
        terrainbento results in writing out the storm sequence file and
        the precipitation exceedence statistics summary if
        ``record_rain`` was set to ``True``.
        """
        # if rain was recorded, write it out.
        if self.record_rain:
            self.write_storm_sequence_to_file(
                filename=self.storm_sequence_filename
            )

            if self.opt_stochastic_duration is False:
                # if opt_stochastic_duration is False, calculate exceedance
                # frequencies and write out.
                try:
                    self.write_exceedance_frequency_file(
                        filename=self.frequency_filename
                    )
                except IndexError:
                    msg = (
                        "terrainbento stochastic model: the rain record was "
                        "too short to calculate exceedance frequency statistics."
                    )
                    os.remove(self.frequency_filename)
                    raise RuntimeError(msg)

    def record_rain_event(
        self, event_start_time, event_duration, rainfall_rate, runoff_rate
    ):
        """Record rain events.

        Create a record of event start time, event duration, rainfall rate, and
        runoff rate.

        Parameters
        ----------
        event_start_time : float
        event_duration : float
        rainfall_rate : float
        runoff_rate : float
        """
        self.rain_record["event_start_time"].append(event_start_time)
        self.rain_record["event_duration"].append(event_duration)
        self.rain_record["rainfall_rate"].append(rainfall_rate)
        self.rain_record["runoff_rate"].append(runoff_rate)

    def write_storm_sequence_to_file(self, filename="storm_sequence.txt"):
        """Write event duration and intensity to a formatted text file.

        Parameters
        ----------
        filename : str
            Default value is "storm_sequence.txt"
        """

        # Open a file for writing
        if self.record_rain is False:
            raise ValueError(
                "Rain was not recorded when the model run. To "
                "record rain, set the parameter 'record_rain'"
                "to True."
            )

        with open(filename, "w") as stormfile:
            stormfile.write(
                "event_start_time"
                + ","
                + "event_duration"
                + ","
                + "rainfall_rate"
                + ","
                + "runoff_rate"
                + "\n"
            )

            n_events = len(self.rain_record["event_start_time"])
            for i in range(n_events):
                stormfile.write(
                    str(
                        np.around(
                            self.rain_record["event_start_time"][i], decimals=5
                        )
                    )
                    + ","
                    + str(
                        np.around(
                            self.rain_record["event_duration"][i], decimals=5
                        )
                    )
                    + ","
                    + str(
                        np.around(
                            self.rain_record["rainfall_rate"][i], decimals=5
                        )
                    )
                    + ","
                    + str(
                        np.around(
                            self.rain_record["runoff_rate"][i], decimals=5
                        )
                    )
                    + "\n"
                )

    def write_exceedance_frequency_file(
        self, filename="exceedance_summary.txt"
    ):
        """Write summary of rainfall exceedance statistics to file.

        Parameters
        ----------
        filename : str
            Default value is "exceedance_summary.txt"
        """
        if self.record_rain is False:
            raise ValueError(
                "Rain was not recorded when the model run. To "
                "record rain, set the parameter 'record_rain'"
                "to True."
            )

        # calculate the number of wet days per year.
        number_of_days_per_year = 365
        nwet = int(
            np.ceil(
                self.rainfall_intermittency_factor * number_of_days_per_year
            )
        )

        if nwet == 0:
            raise ValueError(
                "No rain fell, which makes calculating exceedance "
                "frequencies problematic. We recommend that you "
                "check the valude of rainfall_intermittency_factor."
            )

        with open(filename, "w") as exceedance_file:

            # ndry = int(number_of_days_per_year - nwet)

            # Write some basic information about the distribution to the file.
            exceedance_file.write("Section 1: Distribution Description\n")
            exceedance_file.write(
                "Scale Factor: " + str(self.scale_factor) + "\n"
            )
            exceedance_file.write(
                "Shape Factor: " + str(self.shape_factor) + "\n"
            )
            exceedance_file.write(
                (
                    "Intermittency Factor: "
                    + str(self.rainfall_intermittency_factor)
                    + "\n"
                )
            )
            exceedance_file.write(
                ("Number of wet days per year: " + str(nwet) + "\n\n")
            )
            message_text = (
                "The scale factor that describes this distribution is "
                + "calculated based on a provided value for the mean wet day rainfall."
            )
            exceedance_file.write(
                "\n".join(textwrap.wrap(message_text, _STRING_LENGTH))
            )
            exceedance_file.write("\n")

            exceedance_file.write(
                (
                    "This provided value was:\n"
                    + str(self.rainfall__mean_rate)
                    + "\n"
                )
            )

            # calculate the predictions for 10, 25, and 100 year event based on
            # the analytical form of the exceedance function.
            event_intervals = np.array([10.0, 25, 100.0])

            # calculate the probability of each event based on the number of years
            # and the number of wet days per year.
            daily_distribution_exceedance_probabilities = 1.0 / (
                nwet * event_intervals
            )

            # exceedance probability is given as
            # Probability of daily rainfall of p exceeding a value of po is given as:
            #
            # P(p>po) = e^(-(po/P)^c)
            # P = scale
            # c = shape
            #
            # this can be re-arranged to
            #
            # po = P * (- ln (P(p>po))) ^ (1 / c)

            expected_rainfall = self.scale_factor * (
                -1.0 * np.log(daily_distribution_exceedance_probabilities)
            ) ** (1.0 / self.shape_factor)

            exceedance_file.write("\n\nSection 2: Theoretical Predictions\n")

            message_text = (
                "Based on the analytical form of the wet day rainfall "
                + "distribution, we can calculate theoretical predictions "
                + "of the daily rainfall amounts associated with N-year events."
            )
            exceedance_file.write(
                "\n".join(textwrap.wrap(message_text, _STRING_LENGTH))
            )
            exceedance_file.write("\n")

            for i in range(len(daily_distribution_exceedance_probabilities)):
                exceedance_file.write(
                    (
                        "Expected value for the wet day total of the "
                        + str(event_intervals[i])
                        + " year event is: "
                        + str(np.round(expected_rainfall[i], decimals=3))
                        + "\n"
                    )
                )

            # get rainfall record and filter out time without any rain
            all_precipitation = np.array(self.rain_record["rainfall_rate"])
            rainy_day_inds = np.where(all_precipitation > 0)
            wet_day_totals = all_precipitation[rainy_day_inds]
            num_days = len(wet_day_totals)

            # construct the distribution of yearly maxima.
            # here an effective year is represented by the number of draws implied
            # by the intermittency factor

            # first calculate the number of effective years.

            num_effective_years = int(np.floor(wet_day_totals.size / nwet))

            # write out the calculated event only if the duration
            exceedance_file.write("\n\n")
            message_text = (
                "Section 3: Predicted 95% confidence bounds on the "
                + "exceedance values based on number of samples drawn."
            )
            exceedance_file.write(
                "\n".join(textwrap.wrap(message_text, _STRING_LENGTH))
            )
            exceedance_file.write("\n")

            message_text = (
                "The ability to empirically estimate the rainfall "
                + "associated with an N-year event depends on the "
                + "probability of that event occurring and the number of "
                + "draws from the probability distribution. The ability "
                + "to estimate increases with an increasing number of samples "
                + "and decreases with decreasing probability of event "
                + "occurrence."
            )
            exceedance_file.write(
                "\n".join(textwrap.wrap(message_text, _STRING_LENGTH))
            )
            exceedance_file.write("\n")

            message_text = (
                "Exceedance values calculated from "
                + str(len(wet_day_totals))
                + " draws from the daily-rainfall probability distribution. "
                + "This corresponds to "
                + str(num_effective_years)
                + " effective years."
            )
            exceedance_file.write(
                "\n".join(textwrap.wrap(message_text, _STRING_LENGTH))
            )
            exceedance_file.write("\n")

            # For a general probability distribution, f, with a continuous not zero
            # quantile function at F-1(p), the order statistic associated with the
            # p percentile given n draws from the distribution is given as:

            # X[np] ~ AN ( F-1(p), (p * (p - 1 ))/ (n * [f (F-1 (p)) ]**2))

            # where AN is the asymptotic normal. The value for the variance is more
            # intuitive once you consider that [f (F-1 (p)) ] is the probability
            # that an event of percentile p will occur. Thus the variance increases
            # non-linearly with decreasing event probability and decreases linearly
            # with increaseing observations.

            # we"ve already calculated F-1(p) for our events, and it is represented
            # by the variable expected_rainfall

            daily_distribution_event_percentile = (
                1.0 - daily_distribution_exceedance_probabilities
            )

            event_probability = (
                (self.shape_factor / self.scale_factor)
                * (
                    (expected_rainfall / self.scale_factor)
                    ** (self.shape_factor - 1.0)
                )
                * (
                    np.exp(
                        -1.0
                        * (expected_rainfall / self.scale_factor)
                        ** self.shape_factor
                    )
                )
            )

            event_variance = (
                daily_distribution_event_percentile
                * (1.0 - daily_distribution_event_percentile)
            ) / (num_days * (event_probability ** 2))

            event_std = event_variance ** 0.5

            t_statistic = stats.t.ppf(
                0.975, num_effective_years, loc=0, scale=1
            )

            exceedance_file.write("\n")
            message_text = (
                "For the given number of samples, the 95% "
                + "confidence bounds for the following event "
                + "return intervals are as follows: "
            )
            exceedance_file.write(
                "\n".join(textwrap.wrap(message_text, _STRING_LENGTH))
            )
            exceedance_file.write("\n")
            for i in range(len(event_intervals)):

                min_expected_val = (
                    expected_rainfall[i] - t_statistic * event_std[i]
                )
                max_expected_val = (
                    expected_rainfall[i] + t_statistic * event_std[i]
                )

                exceedance_file.write(
                    (
                        "Expected range for the wet day total of the "
                        + str(event_intervals[i])
                        + " year event is: ("
                        + str(np.round(min_expected_val, decimals=3))
                        + ", "
                        + str(np.round(max_expected_val, decimals=3))
                        + ")\n"
                    )
                )
            # next, calculate the emperical exceedance values, if a sufficient record
            # exists.

            # inititialize a container for the maximum yearly precipitation.
            maximum_yearly_precipitation = np.nan * np.zeros(
                (num_effective_years)
            )
            for yi in range(num_effective_years):

                # identify the starting and ending index coorisponding to the
                # year
                starting_index = yi * nwet
                ending_index = starting_index + nwet

                # select the years portion of the wet_day_totals
                selected_wet_day_totals = wet_day_totals[
                    starting_index:ending_index
                ]

                # record the yearly maximum precipitation
                maximum_yearly_precipitation[
                    yi
                ] = selected_wet_day_totals.max()

            # calculate the distribution percentiles associated with each interval
            event_percentiles = (1.0 - (1.0 / event_intervals)) * 100.0

            # calculated the event magnitudes associated with the percentiles.
            event_magnitudes = np.percentile(
                maximum_yearly_precipitation, event_percentiles
            )

            # write out the calculated event only if the duration
            exceedance_file.write("\n\nSection 4: Empirical Values\n")
            message_text = (
                "These empirical values should be interpreted in the "
                + "context of the expected ranges printed in Section 3. "
                + "If the expected range is large, consider using a longer "
                + "record of rainfall. The empirical values should fall "
                + "within the expected range at a 95% confidence level."
            )
            exceedance_file.write(
                "\n".join(textwrap.wrap(message_text, _STRING_LENGTH))
            )
            exceedance_file.write("\n")

            for i in range(len(event_percentiles)):

                exceedance_file.write(
                    (
                        "Estimated value for the wet day total of the "
                        + str(np.round(event_intervals[i], decimals=3))
                        + " year event is: "
                        + str(np.round(event_magnitudes[i], decimals=3))
                        + "\n"
                    )
                )


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    sm = StochasticErosionModel.from_file(infile)
    sm.run()


if __name__ == "__main__":
    main()
