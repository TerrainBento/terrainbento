#!/usr/bin/env python3

import itertools

from terrainbento.output_writers.generic_output_writer import GenericOutputWriter


class StaticIntervalOutputWriter(GenericOutputWriter):
    r"""Base class for new style output writers or converted old style
    output writers that want to use predetermined output intervals or times.

    The derived class defines what is actually produced. This base class
    handles when output occurs (and interfacing with the model loop via
    GenericOutputWriter).

    At minimum, derived classes must define **run_one_step** for generating the
    actual output. Calling **register_output_filepath** from the derived class
    allows for some optional file management features.

    See constructor and :py:class:`GenericOutputWriter` for more info.
    """

    def __init__(
        self,
        model,
        name="static-interval-output-writer",
        intervals=None,
        intervals_repeat=True,
        times=None,
        **generic_writer_kwargs,
    ):
        """A class for generating output at predetermined intervals or times.

        Parameters
        ----------
        model : a terrainbento ErosionModel instance

        name : string, optional
            The name of the output writer used when generating output
            filenames. Defaults to "static-interval-output-writer".

        intervals : float, int, list of floats, list of ints, optional
            A single float or int value indicates uniform intervals between
            output calls. A list of floats or ints indicates variable
            intervals between output times. Defaults to None which indicates
            that `times` will be used. If both `intervals` and `times` are
            None, will default to the producing one output at the end of the
            model run.

        intervals_repeat : bool, optional
            Indicates whether a list of intervals should repeat until the end
            of the model run. Only has effect if intervals is a list. Has no
            effect for scalar intervals (which always repeat) or if times is
            provided instead of intervals. Default is True.

        times : float, int, list of floats, list of ints, optional
            A single float or int value indicates only one output time.  A list
            of floats or ints indicates multiple predetermined output times.
            Defaults to None which indicates that `intervals` will be used. If
            both `intervals` and `times` are None, will default to the one
            output at the end of the model run. The user must ensure that the
            times implied by `times_iter` align with the model timesteps used
            by the Clock. If a timestep is skipped a warning is raised and if
            more than five timesteps are skipped an error is raised.

        generic_writer_kwargs : keyword args, optional
            Keyword arguments that will be passed directly to
            GenericOutputWriter. These include:

                * add_id : bool, defaults to True
                * save_first_timestep : bool, defaults to False
                * save_last_timestep : bool, defaults to True
                * output_dir : string, defaults to './output'
                * verbose : bool, defaults to False

            Please see :py:class:`GenericOutputWriter` for more detail.

        Returns
        -------
        StaticIntervalOutputWriter: object

        Examples
        --------
        StaticIntervalOutputWriter is a base class that should not be run by
        itself. It contains the machinery for easily creating the output times
        iterator, but does not define **run_one_step**. Please see the
        terrainbento tutorial for output examples.

        """

        super().__init__(
            model,
            name=name,
            **generic_writer_kwargs,
        )
        self._intervals_repeat = intervals_repeat

        # Assert that intervals and times are not both provided. Not clear
        # which one should be used so crash the program.
        # Checks if at least one of the args is None
        assert intervals is None or times is None, "".join(
            [
                "StaticIntervalOutputWriter does not accept both output ",
                "interval and output times simultaneously.",
            ]
        )

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
        # else:
        #   Not a possible scenario. I use elif to be explicit with what that
        #   section is for.

    def _process_intervals_arg(self, intervals):
        """Private method for processing the 'intervals' value provided to the
        constructor.

        Parameters
        ----------
        intervals : float, int, list of floats, list of ints
            A single float or int value indicates uniform intervals between
            output calls. A list of floats or ints indicates variable
            intervals between output times. A list of intervals may be repeated
            if self._intervals_repeat is True.

        Returns
        -------
        times_iter :
            An iterator of floats representing output times.

        """
        times_iter = None
        if isinstance(intervals, (int, float)):
            # 'intervals' is a single integer or float number.
            assert intervals > 0, "Intervals must be positive number(s)"

            # Create a counting iterator that starts at 'intervals' and steps
            # by 'intervals'. Make sure the iterator produces floats
            start = float(intervals)
            step = float(intervals)
            times_iter = (start + step * i for i in itertools.count())
            # times_iter = itertools.count(intervals, intervals)

        elif isinstance(intervals, list):
            if not all(isinstance(i, (int, float)) for i in intervals):
                # The list must contain only floats
                raise NotImplementedError(
                    "".join(
                        [
                            "Only floats or integers are currently supported for ",
                            "the output interval list.",
                        ]
                    )
                )
            assert all(i > 0 for i in intervals), "Intervals must be positive number(s)"

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
        """Private method for processing the 'times' value provided to the
        constructor.

        Parameters
        ----------
        times : float, int, list of floats, list of ints
            A single float or int value indicates only one output time.  A list
            of floats or ints indicates multiple predetermined output times.

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
                raise NotImplementedError(
                    "".join(
                        [
                            "Only floats or integers are currently supported for ",
                            "the output times list.",
                        ]
                    )
                )
            # Create an iterator that steps through the provided list of output
            # times. Make sure they are floats
            # times_iter = iter(times)
            times_iter = (float(i) for i in times)

        else:
            raise NotImplementedError(
                f"Output times type {type(times)} not supported yet."
            )

        assert times_iter is not None
        return times_iter

    # Methods to override
    def run_one_step(self):
        """The function which actually writes data to files (or screen)."""
        # This code is not needed here because it's in GenericOutputWriter...
        # But it's nice for explicitly showing that this function needs to be
        # defined by inheriting classes.
        raise NotImplementedError(
            "The inheriting class needs to implement this function"
        )
