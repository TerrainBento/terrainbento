# -*- coding: utf-8 -*-
"""
stochastic_erosion_model.py: generic base class for an erosion model
that uses stochastic hydrology.

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model import _ErosionModel

from landlab.components import (PrecipitationDistribution)

import numpy as np
import scipy.stats as stats
import textwrap

_STRING_LENGTH = 80

class _StochasticErosionModel(_ErosionModel):
    """
    An StochasticErosionModel is a basic model for erosion and landscape
    evolution in a watershed, as represented by an input DEM.

    This is a base class that handles only processes used by all Stochastic
    Hydrology based modeles.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the _BaseSt base class."""

        # Call _StochasticErosionModel init
        super(_StochasticErosionModel, self).__init__(input_file=input_file,
                                                      params=params,
                                                      BaselevelHandlerClass=BaselevelHandlerClass)

        self.opt_stochastic_duration = (self.params['opt_stochastic_duration'])
        # initialize record for storms. Depending on how this model is run
        # (stochastic time, number_time_steps>1, more manually) the dt may
        # change. Thus, rather than writing routines to reconstruct the time
        # series of precipitation from the dt could change based on users use,
        # we'll record this with the model run instead of re-running.

        # make this the non-default option.

        # First test for consistency between filenames and boolean parameters
        if (((self.params.get('storm_sequence_filename') is not None) or
            (self.params.get('frequency_filename') is not None)) and
                (self.params.get('record_rain') != True)):
            print('A storm sequence or frequency filename was specified but '
                  'record_rain was not set or set to False. Overriding '
                  'record_rain and recording rain so that the file can be '
                  'written')
            self.params['record_rain'] = True

        # Second, test that
        if self.params.get('record_rain'):
            self.record_rain = True
            self.rain_record = {'event_start_time': [],
                                'event_duration': [],
                                'rainfall_rate': [],
                                'runoff_rate': []}
        else:
            self.record_rain = False
            self.rain_record = None

        # check that if (self.opt_stochastic_duration==True) that
        # frequency_filename does not exist. For stochastic time, computing
        # exceedance frequencies is not super sensible. So make a warning that
        # it won't be done.
        if ((self.opt_stochastic_duration==True) and
            (self.params.get('frequency_filename'))):
            print('opt_stochastic_duration is set to True and a '
                  'frequency_filename was specified. Frequency calculations '
                  'are not done with stochastic time so the filename is being '
                  'ignored.')

    def run_for_stochastic(self, dt, runtime):
        """
        Run model without interruption for a specified time period, using
        random storm/interstorm sequence.
        """
        self.rain_generator.delta_t = dt
        self.rain_generator.run_time = runtime
        for (tr, p) in self.rain_generator.yield_storm_interstorm_duration_intensity():
            self.rain_rate = p
            self.run_one_step(tr)

    def instantiate_rain_generator(self):
        """Instantiate RainGenerator."""
        # Handle option for duration.
        self.opt_stochastic_duration = (self.params['opt_stochastic_duration'])
        if self.opt_stochastic_duration:
            self.rain_generator = \
                PrecipitationDistribution(mean_storm_duration=self.params['mean_storm_duration'],
                                          mean_interstorm_duration=self.params['mean_interstorm_duration'],
                                          mean_storm_depth=self.params['mean_storm_depth'],
                                          total_t=self.params['run_duration'],
                                          delta_t=self.params['dt'],
                                          random_seed=int(self.params['random_seed']))
            self.run_for = self.run_for_stochastic  # override base method
        else:
            from scipy.special import gamma
            mean_storm__intensity = (self._length_factor)*self.params['mean_storm__intensity']# has units length per time
            intermittency_factor = self.params['intermittency_factor']

            self.rain_generator = \
                PrecipitationDistribution(mean_storm_duration=1.0,
                                          mean_interstorm_duration=1.0,
                                          mean_storm_depth=1.0,
                                          random_seed=int(self.params['random_seed']))
            self.intermittency_factor = intermittency_factor
            self.mean_storm__intensity = mean_storm__intensity
            self.shape_factor = self.params['precip_shape_factor']
            self.scale_factor = (self.mean_storm__intensity
                                 / gamma(1.0 + (1.0 / self.shape_factor)))
            self.n_sub_steps = int(self.params['number_of_sub_time_steps'])

    def reset_random_seed(self):
        """Re-set the random number generation sequence."""
        try:
            seed = int(self.params['random_seed'])
        except KeyError:
            seed = 0
        self.rain_generator.seed_generator(seedval=seed)

    def handle_water_erosion(self, dt, flooded):
        """Handle water erosion.

           If we are running stochastic duration, then self.rain_rate will
           have been calculated already. It might be zero, in which case we
           are between storms, so we don't do water erosion.

           If we're NOT doing stochastic duration, then we'll run water
           erosion for one or more sub-time steps, each with its own
           randomly drawn precipitation intensity.

           This routine assumes that a model-specific method

                       **calc_runoff_and_discharge()**

           will have been defined.

           For example, BasicStVs calculated runoff and discharge in a different
           way than the other models.

           If the model has a function **update_threshold_field**, this
           function will test for it and run it. This is presently done in
           BasicDdSt.

        """
        # (if we're varying precipitation parameters through time, update them)
        if self.opt_var_precip:
            self.intermittency_factor, self.mean_storm__intensity = self.pc.get_current_precip_params(self.model_time)

        if self.opt_stochastic_duration and self.rain_rate > 0.0:

            runoff = self.calc_runoff_and_discharge()

            self.eroder.run_one_step(dt, flooded_nodes=flooded,
                                     rainfall_intensity_if_used=runoff)
            if self.record_rain:
                #save record into the rain record
                self.record_rain_event(self.model_time, dt, self.rain_rate, runoff)

        elif self.opt_stochastic_duration and self.rain_rate <= 0.0:
            # calculate and record the time with no rain:
            if self.record_rain:
                self.record_rain_event(self.model_time, dt, 0, 0)

        elif not self.opt_stochastic_duration:

            dt_water = ((dt * self.intermittency_factor)
                         / float(self.n_sub_steps))
            for i in range(self.n_sub_steps):
                self.rain_rate = \
                    self.rain_generator.generate_from_stretched_exponential(
                        self.scale_factor, self.shape_factor)

                runoff = self.calc_runoff_and_discharge()
                self.eroder.run_one_step(dt_water, flooded_nodes=flooded,
                                         rainfall_intensity_if_used=runoff)
                #save record into the rain record
                if self.record_rain:
                    event_start_time = self.model_time + (i * dt_water)
                    self.record_rain_event(event_start_time, dt_water, self.rain_rate, runoff)

            # once all the rain time_steps are complete,
            # calculate and record the time with no rain:
            if self.record_rain:

                # calculate dry time
                dt_dry = dt * (1 - self.intermittency_factor)

                # if dry time is greater than zero, record.
                if dt_dry > 0:
                    event_start_time = self.model_time + ((i + 1) * dt_water)
                    self.record_rain_event(event_start_time, dt_dry, 0.0, 0.0)

    def finalize(self):

        # if rain was recorded, write it out.
        if self.record_rain:
            filename = self.params.get('storm_sequence_filename')
            self.write_storm_sequence_to_file(filename)

        if self.record_rain and (self.opt_stochastic_duration==False):
            # if opt_stochastic_duration = False, calculate exceedance
            # frequencies and write out.
            frequency_filename = self.params.get('frequency_filename')
            self.write_exceedance_frequency_file(frequency_filename)

    def record_rain_event(self, event_start_time, event_duration, rainfall_rate, runoff_rate):
        """Record rain events.

        Create a record of event start time, event duration, rainfall rate, and
        runoff rate.

        """
        self.rain_record['event_start_time'].append(event_start_time)
        self.rain_record['event_duration'].append(event_duration)
        self.rain_record['rainfall_rate'].append(rainfall_rate)
        self.rain_record['runoff_rate'].append(runoff_rate)

    def write_storm_sequence_to_file(self, filename=None):
        """
        Write event duration and intensity to a formatted text file.
        """

        # Open a file for writing
        if self.record_rain == False:
            raise ValueError('Rain was not recorded when the model run. To '
                             'record rain, set the parameter "record_rain"'
                             'to True.')
        if filename is None:
            filename = 'event_sequence.txt'
        stormfile = open(filename, 'w')
        stormfile.write('event_start_time' + ',' +
                        'event_duration' + ',' +
                        'rainfall_rate' + ',' +
                        'runoff_rate' + '\n')

        n_events = len(self.rain_record['event_start_time'])
        for i in range(n_events):
            stormfile.write(str(self.rain_record['event_start_time'][i]) + ',' +
                            str(self.rain_record['event_duration'][i]) + ',' +
                            str(self.rain_record['rainfall_rate'][i]) + ',' +
                            str(self.rain_record['runoff_rate'][i])+ '\n')

        # Close the file
        stormfile.close()

    def write_exceedance_frequency_file(self, filename=None):
        """
        """
        if filename is None:
            filename = 'exceedance_summary.txt'
        exceedance_file = open(filename, 'w')

        # calculate the number of wet days per year.
        number_of_days_per_year = 365
        nwet = int(np.ceil(self.intermittency_factor * number_of_days_per_year))
        #ndry = int(number_of_days_per_year - nwet)

        # Write some basic information about the distribution to the file.
        exceedance_file.write('Section 1: Distribution Description\n')
        exceedance_file.write('Scale Factor: ' + str(self.scale_factor) + '\n')
        exceedance_file.write('Shape Factor: ' + str(self.shape_factor) + '\n')
        exceedance_file.write(('Intermittency Factor: ' +
                               str(self.intermittency_factor) + '\n'))
        exceedance_file.write(('Number of wet days per year: ' +
                               str(nwet) + '\n\n'))
        message_text = ('The scale factor that describes this distribution is ' +
                        'calculated based on a provided value for the mean wet day rainfall.')
        exceedance_file.write('\n'.join(textwrap.wrap(message_text, _STRING_LENGTH)))
        exceedance_file.write('\n')

        exceedance_file.write(('This provided value was:\n' +
                               str(self.mean_storm__intensity) + '\n'))

        # calculate the predictions for 10, 25, and 100 year event based on
        # the analytical form of the exceedance function.
        event_intervals = np.array([10., 25, 100.])

        # calculate the probability of each event based on the number of years
        # and the number of wet days per year.
        daily_distribution_exceedance_probabilities = (1./(nwet * event_intervals))

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

        expected_rainfall = self.scale_factor * (-1. * np.log(daily_distribution_exceedance_probabilities)) ** (1. / self.shape_factor)

        exceedance_file.write('\n\nSection 2: Theoretical Predictions\n')

        message_text = ('Based on the analytical form of the wet day rainfall ' +
                        'distribution, we can calculate theoretical predictions ' +
                        'of the daily rainfall amounts associated with N-year events.')
        exceedance_file.write('\n'.join(textwrap.wrap(message_text, _STRING_LENGTH)))
        exceedance_file.write('\n')

        for i in range(len(daily_distribution_exceedance_probabilities)):
            exceedance_file.write(('Expected value for the wet day total of the ' +
                                   str(event_intervals[i]) +
                                   ' year event is: ' +
                                   str(np.round(expected_rainfall[i], decimals=3)) + '\n'))

        # get rainfall record and filter out time without any rain
        all_precipitation = np.array(self.rain_record['rainfall_rate'])
        rainy_day_inds = np.where(all_precipitation>0)
        if len(rainy_day_inds[0])>0:
            wet_day_totals = all_precipitation[rainy_day_inds]
        else:
            raise ValueError('No rain fell, which makes calculating exceedance '
                             'frequencies problematic. We recommend that you '
                             'check the valude of intermittency_factor.')

        # construct the distribution of yearly maxima.
        # here an effective year is represented by the number of draws implied
        # by the intermittency factor

        # first calculate the number of effective years.
        num_days = len(wet_day_totals)
        num_effective_years = int(np.floor(wet_day_totals.size/nwet))


        # write out the calculated event only if the duration
        exceedance_file.write('\n\n')
        message_text = ('Section 3: Predicted 95% confidence bounds on the ' +
                        'exceedance values based on number of samples drawn.')
        exceedance_file.write('\n'.join(textwrap.wrap(message_text, _STRING_LENGTH)))
        exceedance_file.write('\n')

        message_text = ('The ability to empirically estimate the rainfall ' +
                        'associated with an N-year event depends on the ' +
                        'probability of that event occurring and the number of ' +
                        'draws from the probability distribution. The ability ' +
                        'to estimate increases with an increasing number of samples ' +
                        'and decreases with decreasing probability of event ' +
                        'occurrence.')
        exceedance_file.write('\n'.join(textwrap.wrap(message_text, _STRING_LENGTH)))
        exceedance_file.write('\n')

        message_text = ('Exceedance values calculated from ' + str(len(wet_day_totals)) +
                        ' draws from the daily-rainfall probability distribution. '+
                        'This corresponds to ' + str(num_effective_years) +
                        ' effective years.')
        exceedance_file.write('\n'.join(textwrap.wrap(message_text, _STRING_LENGTH)))
        exceedance_file.write('\n')

        # For a general probability distribution, f, with a continuous not zero
        # quantile function at F-1(p), the order statistic associated with the
        # p percentile given n draws from the distribution is given as:

        # X[np] ~ AN ( F-1(p), (p * (p - 1 ))/ (n * [f (F-1 (p)) ]**2))

        # where AN is the asymptotic normal. The value for the variance is more
        # intuitive once you consider that [f (F-1 (p)) ] is the probability
        # that an event of percentile p will occur. Thus the variance increases
        # non-linearly with decreasing event probability and decreases linearly
        # with increaseing observations.

        # we've already calculated F-1(p) for our events, and it is represented
        # by the variable expected_rainfall

        daily_distribution_event_percentile = 1.0 - daily_distribution_exceedance_probabilities

        event_probability = ((self.shape_factor/self.scale_factor) *
                             ((expected_rainfall/self.scale_factor) ** (self.shape_factor - 1.0)) *
                             (np.exp(-1. * (expected_rainfall/self.scale_factor) ** self.shape_factor)))

        event_variance = ((daily_distribution_event_percentile * (1.0 - daily_distribution_event_percentile)) /
                          (num_days * (event_probability ** 2)))

        event_std = event_variance ** 0.5

        t_statistic = stats.t.ppf(0.975, num_effective_years, loc=0, scale=1)

        exceedance_file.write('\n')
        message_text = ('For the given number of samples, the 95% ' +
                        'confidence bounds for the following event ' +
                        'return intervals are as follows: ')
        exceedance_file.write('\n'.join(textwrap.wrap(message_text, _STRING_LENGTH)))
        exceedance_file.write('\n')
        for i in range(len(event_intervals)):

            min_expected_val = expected_rainfall[i] - t_statistic * event_std[i]
            max_expected_val = expected_rainfall[i] + t_statistic * event_std[i]

            exceedance_file.write(('Expected range for the wet day total of the ' +
                                   str(event_intervals[i]) +
                                   ' year event is: (' +
                                   str(np.round(min_expected_val, decimals=3)) + ', ' +
                                   str(np.round(max_expected_val, decimals=3)) + ')\n'))
        # next, calculate the emperical exceedance values, if a sufficient record
        # exists.

        # inititialize a container for the maximum yearly precipitation.
        maximum_yearly_precipitation = np.nan * np.zeros((num_effective_years))
        for yi in range(num_effective_years):

            # identify the starting and ending index coorisponding to the
            # year
            starting_index = yi*nwet
            ending_index = starting_index+nwet

            # select the years portion of the wet_day_totals
            selected_wet_day_totals = wet_day_totals[starting_index:ending_index]

            # record the yearly maximum precipitation
            maximum_yearly_precipitation[yi] = selected_wet_day_totals.max()


        # calculate the distribution percentiles associated with each interval
        event_percentiles = (1. - (1./event_intervals)) * 100.

        # calculated the event magnitudes associated with the percentiles.
        event_magnitudes = np.percentile(maximum_yearly_precipitation, event_percentiles)

        # write out the calculated event only if the duration
        exceedance_file.write('\n\nSection 4: Empirical Values\n')
        message_text = ('These empirical values should be interpreted in the ' +
                        'context of the expected ranges printed in Section 3. ' +
                        'If the expected range is large, consider using a longer ' +
                        'record of rainfall. The empirical values should fall ' +
                        'within the expected range at a 95% confidence level.')
        exceedance_file.write('\n'.join(textwrap.wrap(message_text, _STRING_LENGTH)))
        exceedance_file.write('\n')

        for i in range(len(event_percentiles)):

            exceedance_file.write(('Estimated value for the wet day total of the ' +
                                   str(np.round(event_intervals[i], decimals=3)) +
                                   ' year event is: ' +
                                   str(np.round(event_magnitudes[i], decimals=3)) + '\n'))

        exceedance_file.close()


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    em = _StochasticErosionModel(input_file=infile)
    em.run()


if __name__ == '__main__':
    main()
