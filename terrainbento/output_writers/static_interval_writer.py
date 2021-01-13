#!/usr/bin/env python3

import itertools
from terrainbento.output_writers import GenericOutputWriter

class StaticIntervalOutputWriter(GenericOutputWriter):
    def __init__(self, 
            model, 
            name="static-interval-output-writer",
            add_id=True,
            save_first_timestep=False,
            intervals=None,
            intervals_repeat=False,
            times=None
            ):
        """A class for generating output at predetermined intervals or times.

        Parameters
        ----------
        model : a terrainbento ErosionModel instance
        
        name : string, optional
            The name of the output writer used when generating output 
            filenames. Defaults to "static-interval-output-writer".

        add_id : bool, optional
            Indicates whether to append "-{id}" to the end of the name. False 
            does nothing. Default is True.

        save_first_timestep : bool, optional
            Indicates that the first output time is at time zero. Defaults to 
            False.

        intervals : float, list of floats, optional
            A single float value indicates uniform intervals between output 
            calls. A list of floats indicates predetermined intervals between 
            output times. Defaults to None which indicates that `times` will be 
            used.  If both `intervals` and `times` are None, will default to 
            the producing one output at the end of the model run.

        intervals_repeat : bool, optional
            Indicates whether a list of intervals should repeat until the end 
            of the model run. Default is False. Only has effect if intervals is 
            a list. Has no effect for scalar intervals (which always 
            repeat) or if times is provided instead of intervals.

        times : list of floats, optional
            A list of model times to generate output. Either `intervals` or 
            `times` should be defined. Defaults to None which indicates that 
            `intervals` will be used. If both `intervals` and `times` are None, 
            will default to the one output at the end of the model run.

        Returns
        -------

        """

        super().__init__(model, name=name, add_id=add_id, 
                save_first_timestep=save_first_timestep,
                )
        self._intervals_repeat = intervals_repeat

        # Assert that intervals and times are not both provided. Not clear 
        # which one should be used so crash the program.
        # Checks if at least one of the args is None
        assert intervals is None or times is None, ''.join([
                "StaticIntervalOutputWriter does not accept both output ",
                "interval and output times simultaneously.",
                ])
        
        # Check if both args are None
        if intervals is None and times is None:
            # The caller provided neither intervals or times. Use the model end 
            # time instead, meaning there will only be one output at 
            # the end.
            times = model.clock.stop

        # Generate a iterator of output times either indirectly from the output 
        # intervals or directly from output model times.
        if intervals is not None:
            times_iter = self._process_intervals_arg(intervals)
            self.register_times_iter(times_iter)
        elif times is not None:
            times_iter = self._process_times_arg(times)
            self.register_times_iter(times_iter)
        #else:
        #   Not a possible scenario. I use elif to be explicit with what that 
        #   section is for.

    def _process_intervals_arg(self, intervals):
        """ Process the 'intervals' value provided to the constructor.

        Parameters
        ----------
        intervals : float, int , list of floats, list of ints
            A single float/integer value indicates uniform intervals between 
            output calls. A list of floats/ints indicates predetermined 
            intervals between output times.

        Returns
        -------
        times_iter : 
            An iterator of float output times.

        """
        times_iter = None
        if isinstance(intervals, (int, float)):
            # 'intervals' is a single integer or float number.
            #
            # Create a counting iterator that starts at 'intervals' and steps 
            # by 'intervals'. Make sure the iterator produces floats
            start = float(intervals)
            step = float(intervals)
            times_iter = (start + step * i for i in itertools.count())
            #times_iter = itertools.count(intervals, intervals)

        elif isinstance(intervals, list):
            if not all(isinstance(i, (int, float)) for i in intervals):
                # The list must contain only floats
                raise NotImplementedError(''.join([
                    f"Only floats or integers are currently supported for ",
                    f"the output interval list.",
                    ]))

            if self._intervals_repeat:
                # The intervals list should repeat until the end of the model 
                # run. Creates an infinite iterator.
                raw_iter = itertools.accumulate(itertools.cycle(intervals))
                times_iter = (float(i) for i in raw_iter)

            else:
                # Create an iterator that steps through a list of output times 
                # calculated by accumulating the list of output intervals. Make 
                # sure they are floats.
                times_iter = (float(i) for i in itertools.accumulate(intervals))


        else:
            raise NotImplementedError(
                    f"Interval type {type(intervals)} not supported yet."
                    )

        assert times_iter is not None
        return times_iter
    def _process_times_arg(self, times):
        """ Process the 'times' value provided to the constructor.

        Parameters
        ----------
        times : float, int, list of floats, list of ints
            A list of model times to generate output.
            A single float/integer value indicates that the model produces 
            output once at the specified model time. A list of floats/ints 
            indicates multiple predetermined model times to produce 
            output.

        Returns
        -------
        times_iter : 
            An iterator of float output times.

        """
        times_iter = None

        if isinstance(times, (int, float)):
            # If a single integer or float is provided, create an iterator with 
            # only one float value.
            times_iter = iter([float(times)])

        elif isinstance(times, list):
            # Assert that it is a list of floats
            if not all(isinstance(i, (int, float)) for i in times):
                # The list must contain only floats
                raise NotImplementedError(''.join([
                    f"Only floats or integers are currently supported for ",
                    f"the output times list.",
                    ]))
            # Create an iterator that steps through the provided list of output 
            # times. Make sure they are floats
            #times_iter = iter(times)
            times_iter = (float(i) for i in times)

        else:
            raise NotImplementedError(
                    f"Output times type {type(times)} not supported yet."
                    )

        assert times_iter is not None
        return times_iter


    def run_one_step(self):
        r""" The function which actually writes data to files (or screen).
        """
        raise NotImplementedError(
                "The inheriting class needs to implement this function"
                )


