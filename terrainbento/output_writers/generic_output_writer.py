#!/usr/bin/env python3

import itertools
import os
import warnings


class OutputIteratorSkipWarning(UserWarning):
    """
    A UserWarning child class raised when the advancing iterator skips a
    time between zero and the stop time.
    """

    def get_message(next_time, prev_time):
        return "".join(
            [
                f"Next output time {next_time} is less than or equal to the ",
                f"previous output time {prev_time}. Skipping...",
            ]
        )


class GenericOutputWriter:
    r"""Base class for all new style output writers or converted old style
    output writers.

    The derived class defines when output occurs via an iterator and what is
    actually produced. This base class handles the interfacing with the model
    loop.

    At minimum, derived classes must define **run_one_step** for generating the
    actual output and must provide an iterator of output times via either the
    constructor or **register_times_iter**.  Calling
    **register_output_filepath** from the derived class allows for some
    optional file management features.

    See constructor for more details.
    """

    # Generate unique output writer ID numbers
    _id_iter = itertools.count()

    def __init__(
        self,
        model,
        name=None,
        add_id=True,
        save_first_timestep=False,
        save_last_timestep=True,
        output_dir=None,
        times_iter=None,
        verbose=False,
    ):
        r"""Base class for all new style output writers.

        Parameters
        ----------
        model : terrainbento ErosionModel instance

        name : string, optional
            The name of the output writer used for identifying the writer and
            generating file names. Defaults to "output-writer" or
            "output-writer-id{id}" depending on **add_id** argument.

        add_id : bool, optional
            Indicates whether the output writer ID number should be appended to
            the name using the format "-id{id}". Useful if there are
            multiple output writers of the same type with non-unique names.
            Defaults to True.

        save_first_timestep : bool, optional
            Indicates that the first output time must be at time zero
            regardless of whether or not the output time iterator generates
            zero. Defaults to False.

        save_last_timestep : bool, optional
            Indicates that the last output time must be at the clock stop time
            regardless of whether or not the output time iterator would
            normally generate the stop time. Defaults to True.

        output_dir : string, optional
            Directory where output files will be saved. Default value is None,
            which creates an 'output' directory in the current directory.

        times_iter : iterator of floats, optional
            The user can provide an iterator of floats representing output
            times here instead of registering one later using
            **register_times_iter**. The user must ensure that the times
            implied by `times_iter` align with the model timesteps used by the
            Clock. If a timestep is skipped a warning is raised and if more
            than five timesteps are skipped an error is raised.

        Returns
        -------
        GenericOutputWriter: object

        Examples
        --------
        GenericOutputWriter is a base class that should not be run by itself.
        Please see the terrainbento tutorial for output examples.
        """

        self._model = model
        self._save_first_timestep = save_first_timestep
        self._save_last_timestep = save_last_timestep
        self._verbose = verbose

        # Make sure the model has a clock. All models should have clock, but
        # just in case...
        assert (
            hasattr(self.model, "clock") and self.model.clock is not None
        ), "Output writers require that the model has a clock."

        # Generate the id number for this instance
        self._id = next(GenericOutputWriter._id_iter)

        # Generate a default name if necessary
        self._name = name or "output-writer"
        self._name += f"-id{self._id}" if add_id else ""

        # Generate an iterator of output times
        # Needs to be set by register_times_iter
        self._times_iter = None

        # Some variables to track the state of the iterator
        self._next_output_time = None
        self._prev_output_time = None
        self._is_exhausted = False

        # File management
        if output_dir is None:
            # Make a subdir with a some kind of model run identifier?
            # e.g. time stamp for model start time
            output_dir = os.path.join(os.curdir, "output")
        if not os.path.isdir(output_dir):  # pragma: no cover
            self.vprint("Making output directory at {output_dir}")
            os.mkdir(output_dir)
        self._output_dir = output_dir
        self._output_filepaths = []

        # Register the times_iter if one was provided.
        if times_iter is not None:
            self.register_times_iter(times_iter)

    # Attributes
    @property
    def model(self):
        """The model reference."""
        return self._model

    @property
    def id(self):
        """The output writer's unique id number."""
        return self._id

    @property
    def name(self):
        """The output writer's name."""
        return self._name

    @property
    def filename_prefix(self):
        """Generate a filename prefix based on the model prefix, writer's
        name, and model time. e.g. model-prefix_ow-name_time-0000000001.0"""

        # Note, model iteration is NOT the number of steps... It is the number
        # of times the run_for loop is executed. Can't use it here even though
        # an integer would be cleaner.
        model_prefix = self.model.output_prefix
        # iteration_str = f"iter-{self.model.iteration:05d}"
        time_str = f"time-{self.model.model_time:012.1f}"  # .replace('.', 'x')
        if model_prefix:
            # prefix = '_'.join([model_prefix, self._name, iteration_str])
            prefix = "_".join([model_prefix, self._name, time_str])
        else:
            # prefix = '_'.join([self._name, iteration_str])
            prefix = "_".join([self._name, time_str])
        return prefix

    @property
    def output_dir(self):
        """Output directory"""
        return self._output_dir

    @property
    def next_output_time(self):
        r"""Return when this object is next supposed to write output. Does NOT
        advance the iterator."""
        return self._next_output_time

    @property
    def prev_output_time(self):
        r"""Returns the previous valid output time. Does not change after the
        time iterator is exhausted."""
        return self._prev_output_time

    @property
    def output_filepaths(self):
        """Return a list of all output filepaths that have been written by
        this writer and registered with **register_output_filepath**."""
        return self._output_filepaths

    # Time iterator methods
    def register_times_iter(self, times_iter):
        """Function for registering an iterator of output times.

        The inheriting class must call this function or provide the iterator to
        the constructor (which then calls this function). This function does
        not check the values in the iterator, but **advance_iter** will.


        Parameters
        ----------
        times_iter : iterator of floats
            An iterator of floats representing model times when the output
            writer should create output. The iterator values should be
            monotonically increasing and non-negative, but there is some
            flexibility in **advance_iter** to skip bad values.
        """

        self._times_iter = times_iter

    def advance_iter(self):
        r"""Public-facing function for advancing the output times iterator.

        The advancing iterator accounts for forced saving on the first/last
        steps and accounts for short sequences where the generated times are
        smaller than the previous value.

        Warnings are thrown when a time between zero and the stop time is
        skipped and a RecursionError is thrown if too many values are skipped
        (default is 5 skips max).

        Returns
        -------
        next_output_time : float or None
            A float value for the next model time when this output writer needs
            to write output. None indicates that this writer has finished
            writing output for the rest of the model run.
        """

        # Assert that the iterator exists and has the next function
        assert (
            self._times_iter is not None
        ), "An output time iterator has not been registered!."
        assert hasattr(
            self._times_iter, "__next__"
        ), "The output time iterator needs a __next__ function."

        # Check if the writer is already in an exhausted state
        if self._is_exhausted:
            # Already exhausted. Always return None
            assert self._next_output_time is None
            return None

        # Writer is not exhausted yet
        had_next = self._next_output_time is not None
        had_prev = self._prev_output_time is not None
        save_first = self._save_first_timestep
        model_stop_time = self.model.clock.stop

        # Update the previous value before advancing the iterator
        if had_next:
            # Only updates the previous time while the iterator is running.
            # (eventually becomes the final valid output time)
            assert self._next_output_time <= model_stop_time
            self._prev_output_time = self._next_output_time

        # Check if the last output time was the stop time.
        if had_next and self._next_output_time == model_stop_time:
            # Previous time was the final step and output was forced by
            # _save_last_timestep. The times iterator was still advanced during
            # the last step and might be returning garbage if used again.
            # e.g. [1,2,3,40,15] with stop time of 20 and save_last_step = True
            # might attempt to write output at t=15.
            next_time = None
        elif save_first and not had_prev and not had_next:
            # First time advancing the iterator (both prev and next are None),
            # but the first output time needs to be at time zero. Set the next
            # time to zero.
            next_time = 0.0
        else:
            # Advance the iterator
            next_time = self._advance_iter_recursive()

        # Check if the writer has become exhausted
        if next_time is None:
            self._is_exhausted = True

        # Save and return the next time
        self._next_output_time = next_time

        return next_time

    def _advance_iter_recursive(self, recursion_counter=5):
        r"""Private function for advancing the output times iterator.

        This function accounts for iterator exhaustion, saving the last time
        step, and values that are smaller than the previous value.

        Recursion is used to skip times that are too small compared to the
        previous output time. Warnings are thrown whenever a time between zero
        and the stop time is skipped and a RecursionError is thrown if too many
        values are skipped (default is 5 skips in a row).

        Parameters
        ----------
        recursion_counter : int, optional
            A counter to track the depth of recursion when skipping values less
            than or equal to the previous value. Defaults to a max depth of 5.

        Returns
        -------
        next_output_time : float or None
            A float value for the next model time when this output writer needs
            to write output. None indicates that this writer has finished
            writing output for the rest of the model run.

        """

        # Advance the time iterator to get the next time value
        next_time = next(self._times_iter, None)
        prev_time = self._prev_output_time  # Already updated by advance_iter()
        model_stop_time = self.model.clock.stop

        if next_time is None:
            # The iterator returned None and is therefore exhausted.

            if self._save_last_timestep:
                # Make sure the last output time will be at the end of the
                # model run.
                if prev_time is None or prev_time < model_stop_time:
                    # The iterator either had no entries or the previous output
                    # time is before model stop time. Either way, make sure the
                    # next output time is the model stop time.
                    return model_stop_time
                # else prev_time >= stop_time -> already output at stop time

            # Output at the model stop time was not required or already
            # occurred. No further times necessary.
            return None

        # For the following code, we know next_time is not None

        # Check that the iterator returned a proper value
        assert isinstance(
            next_time, float
        ), "The output time iterator needs to generate float values."

        if next_time > model_stop_time:
            # The next time is greater than the model end time and should be
            # exhausted. The iterator is too long (most likely infinite) and
            # the interval either jumped over the model stop time or this is
            # the final time step.

            if self._save_last_timestep:
                # Make sure the last output time will be at the end of the
                # model run.
                if prev_time is None or prev_time < model_stop_time:
                    # The iterator jumped past the end time from either the
                    # first advance (i.e. output interval > model duration) or
                    # from a normal advance.  Either way, make sure the next
                    # output time is the model stop time.
                    return model_stop_time
                # else prev_time >= stop_time -> already output at stop time

            # Output at the model stop time was not required or already
            # occurred. No further times necessary.
            return None

        elif (prev_time is not None) and (prev_time >= next_time):
            # Next time is smaller than previous time. Ignore this value and
            # try advancing again until a larger value is found or the
            # recursion_counter runs out.
            if recursion_counter > 0:
                if not (prev_time == 0 and next_time == 0):
                    # Warn the user that there are issues with the iterator.
                    # Ignore when time == zero because that may be common when
                    # trying to save the first time step.
                    warning_cls = OutputIteratorSkipWarning
                    warning_msg = warning_cls.get_message(next_time, prev_time)
                    warnings.warn(warning_msg, warning_cls)

                return self._advance_iter_recursive(recursion_counter - 1)
            else:
                raise RecursionError("Too many output times skipped.")
        else:
            # Normal value. Return as is.
            return next_time

    # Methods to override
    def run_one_step(self):
        r"""The function which actually writes data to files or the screen."""
        raise NotImplementedError(
            "The inheriting class needs to implement this function."
        )

    # File management
    def make_filepath(self, filename):
        """Join the output directory to a filename."""
        return os.path.join(self.output_dir, filename)

    def is_file_registered(self, filepath):
        """Check if an output filepath has already been registered with
        this writer.

        Parameters
        ----------
        filepath : string
            Filepath to check.

        Returns
        -------
        is_registered : bool
            True means that the file is already registered. False means file is
            not registered yet.
        """
        return filepath in self._output_filepaths

    def register_output_filepath(self, filepath):
        """Register the filepath to a newly created file.

        Does not throw any errors or warnings if the file is already registered
        or exists. (Should it? User could be intentionally overwriting a file.)

        NOTE: Old style output writers do not have the ability to register
        files. Therefore file registering/management can't be a required
        feature.

        Parameters
        ----------
        filepath : string
            Filepath to a new file that will be registered.
        """

        if not self.is_file_registered(filepath):
            self.vprint(f"Registering a new filepath {filepath}")
            self._output_filepaths.append(filepath)

    def delete_output_files(self, only_extension=None):
        """Delete output files generated by this writer that have been
        registered. Primarily for testing cleanup.

        Parameters
        ----------
        only_extension : string, optional
            Specify what type of files to delete. Defaults to None, which will
            delete all file types generated by this writer that have been
            registered.
        """

        output_filepaths = self._output_filepaths
        keep_filepaths = []

        self.vprint("Deleting files...")
        self.vprint(f"{self.name} wrote: {output_filepaths}")
        for filepath in output_filepaths:
            # Note: ''[1:] will return '' (i.e. does not crash if no extension)
            file_ext = os.path.splitext(filepath)[1][1:]
            if only_extension is None or file_ext in only_extension:
                # Deleting all files or just the target extension type
                self.vprint(f"Deleting {filepath}")
                try:
                    os.remove(filepath)
                except OSError:  # pragma: no cover
                    print(
                        "The Windows OS is picky about file-locks and did "
                        "not permit terrainbento to remove the netcdf files."
                    )
                    keep_filepaths.append(filepath)  # could not delete
            else:
                self.vprint(f"Keeping {filepath}")
                # Not deleting this file
                keep_filepaths.append(filepath)

        self._output_filepaths = keep_filepaths

    def get_output_filepaths(self, only_extension=None):
        """Get a list of all output files created by this writer that have
        been registered.

        Parameters
        ----------
        only_extension : string, optional
            Specify what type of files to return. Defaults to None, which will
            return all file types generated by this writer that have been
            registered.

        Returns
        -------
        filepaths : list of strings
            List of filepath strings that match extension requirements and were
            registered.
        """

        output_filepaths = self._output_filepaths
        return_filepaths = []

        for filepath in output_filepaths:
            # Note: ''[1:] will return '' (i.e. does not crash if no extension)
            file_ext = os.path.splitext(filepath)[1][1:]
            if only_extension is None or file_ext == only_extension:
                return_filepaths.append(filepath)

        return return_filepaths

    def vprint(self, msg):
        """Print output to the standard output stream if in verbose mode."""
        if self._verbose:
            print(msg)
