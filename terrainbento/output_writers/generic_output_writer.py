#!/usr/bin/env python3

import itertools
import warnings

class OutputIteratorSkipWarning(UserWarning):
    """
    A UserWarning child class raised when the advancing iterator skips a 
    non-zero time.
    """
    def get_message(next_time, prev_time):
        return ''.join([
                f"Next output time {next_time} is <= ",
                f"previous output time {prev_time}. Skipping..."
                ])

#warnings.simplefilter('always', OutputIteratorSkipWarning)

class GenericOutputWriter:
    # Generate unique output writer ID numbers
    _id_iter = itertools.count()
    
    def __init__(self, 
            model, 
            name=None, 
            add_id=True, 
            save_first_timestep=False,
            save_last_timestep=True,
            ):
        r"""

        Parameters
        ----------
        model : terrainbento ErosionModel instance

        name : string, optional
            The name of the output writer used for generating file names. 
            Defaults to "output-writer"

        add_id : bool, optional
            Indicates whether the output writer ID number should be appended to 
            the name following the format f"-id{id}". Useful if there are 
            multiple output writers of the same type with non-unique names.  
            Defaults to True.

        save_first_timestep : bool, optional
            Indicates that the first output time is at time zero. Defaults to 
            False.

        save_last_timestep : bool, optional
            Indicates that the last output time is at the clock stop time even 
            if the iterator is infinite or exhausted. 
            Defaults to True.

        [section name?]
        ---------------
        Important! The inheriting class needs to register an iterator of output 
        times by calling `register_times_iter`.

        """

        self.model = model
        self._save_first_timestep = save_first_timestep
        self._save_last_timestep = save_last_timestep

        # Make sure the model has a clock. All models should have clock, but 
        # just in case...
        assert hasattr(self.model, 'clock') and self.model.clock is not None, \
                f"Output writers require that the model has a clock."

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
    
    # Attributes
    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def next_output_time(self):
        r"""
        Return when this object is next supposed to write output.
        Does NOT advance the iterator.
        """
        return self._next_output_time

    @property
    def prev_output_time(self):
        r"""
        Returns the previous valid output time. Does not change after the time 
        iterator is exhausted.
        """
        return self._prev_output_time


    # Time iterator methods
    def register_times_iter(self, times_iter):
        """ Function for registering an iterator of output times.

        The inheriting class must call this function. This function does not 
        check the values in the iterator, but the `write_output` function will.
        

        Parameters
        ----------
        times_iter : iterator of floats
            An iterator of floats representing model times when the output 
            writer should create output.
        """

        self._times_iter = times_iter

    def advance_iter(self):
        r""" Advances the output times iterator.

        Times that are too small compared to the previous output time are 
        skipped. Warnings are thrown when a non-zero time is skipped and a 
        RecursionError is thrown if too many values are skipped (default is 5 
        skips). 

        Returns
        -------
        next_output_time : float or None
            A float value for the next model time when this output writer needs 
            to write output. None indicates that this writer has finished 
            writing output for the rest of the model run.
        """
        
        # Assert that the iterator exists and has the next function
        assert self._times_iter is not None, \
                f"An output time iterator has not been registered!."
        assert hasattr(self._times_iter, '__next__'), \
                f"The output time iterator needs a __next__ function"

        # Check if the writer is already in an exhausted state
        if self._is_exhausted:
            # Already exhausted. Always return None
            assert self._next_output_time == None
            return None
        
        # Writer is not exhausted yet
        had_next = self._next_output_time is not None
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
            # save_last_timestep. The times iterator was still advanced during 
            # the last step and might be returning garbage if used again.
            # e.g. [1,2,3,40,15] with stop time of 20 and save_last_step = True 
            # might attempt to write output at t=15.
            next_time = None
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
        r""" Advances the output times iterator.
        
        Recursion is used to skip times that are too small compared to the 
        previous output time. Warnings are thrown whenever a non-zero time is 
        skipped and a RecursionError is thrown if too many values are skipped 
        (default is 5 skips in a row). 

         Skipping allows some ability to handle a poorly constructed times_iter 
         and the special case of outputting the initial conditions.

        Checks if the current model time is actually the correct time?

        After writing the output, advance the times_iter iterator and return 
        when this object is next supposed to write output.
        
        Parameters
        ----------
        recursion_counter : int
            A counter to track the depth of recursion when skipping values less 
            than or equal to the previous value.

        Returns
        -------
        next_output_time : float or None
            A float value for the next model time when this output writer needs 
            to write output. None indicates that this writer has finished 
            writing output for the rest of the model run.
        """

        if self._save_first_timestep:
            # First time advancing the iterator, but the first output time 
            # needs to be at time zero. Return zero instead of calling next on 
            # the times iterator.
            self._save_first_timestep = False
            return 0.0
        
        # Advance the time iterator to get the next time value
        next_time = next(self._times_iter, None)
        prev_time = self._prev_output_time # Already updated by advance_iter()
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
        assert isinstance(next_time, float), \
                "The output time iterator needs to generate float values."

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

    # Base class method (must be overridden)
    def run_one_step(self):
        r""" The function which actually writes data to files or screen. """
        raise NotImplementedError(
                "The inheriting class needs to implement this function"
                )


