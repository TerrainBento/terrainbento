#! /usr/bin/env python
"""Interface to the basic control functions of a model."""


class BmiBase(object):

    """Functions that control model execution.

    These BMI functions are critical to plug-and-play modeling because they
    give a calling component fine-grained control over the model execution.
    """

    def initialize(self, filename):
        """Perform startup tasks for the model.

        Perform all tasks that take place before entering the model's time
        loop, including opening files and initializing the model state. Model
        inputs are read from a text-based configuration file, specified by
        `filename`.

        Parameters
        ----------
        filename : str, optional
          The path to the model configuration file.

        Notes
        -----
        Models should be refactored, if necessary, to use a
        configuration file. CSDMS does not impose any constraint on
        how configuration files are formatted, although YAML is
        recommended. A template of a model's configuration file
        with placeholder values is used by the BMI.

        .. code-block:: c

            /* C */
            int initialize(void *self, char * filename);
        """
        pass

    def update(self):
        """Advance model state by one time step.

        Perform all tasks that take place within one pass through the model's
        time loop. This typically includes incrementing all of the model's
        state variables. If the model's state variables don't change in time,
        then they can be computed by the :func:`initialize` method and this
        method can return with no action.

        Notes
        -----
        .. code-block:: c

            /* C */
            int update(void *self);
        """
        pass

    def update_until(self, time):
        """Advance model state until the given time.

        Parameters
        ----------
        time : float
          A model time value.

        See Also
        --------
        update

        Notes
        -----
        .. code-block:: c

            /* C */
            int update_until(void *self, double time);
        """
        pass

    def update_frac(self, time_frac):
        """Advance model state by a fraction of a time step.

        Parameters
        ----------
        time_frac : float
          A fraction of a model time step value.

        See Also
        --------
        update

        Notes
        -----
        .. code-block:: c

            /* C */
            int update_frac(void *self, double time_frac);
        """
        pass

    def finalize(self):
        """Perform tear-down tasks for the model.

        Perform all tasks that take place after exiting the model's time
        loop. This typically includes deallocating memory, closing files and
        printing reports.

        Notes
        -----
        .. code-block:: c

            /* C */
            int finalize(void *self);
        """
        pass
