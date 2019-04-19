#! /usr/bin/env python
"""Interface that describes the time stepping of a model."""


class BmiTime(object):

    """Methods that get time information from a model."""

    def get_start_time(self):
        """Start time of the model.

        Model times should be of type float. The default model start
        time is 0.

        Returns
        -------
        float
          The model start time.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_start_time(void * self, double * time);
        """
        pass

    def get_current_time(self):
        """Current time of the model.

        Returns
        -------
        float
          The current model time.

        See Also
        --------
        get_start_time

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_current_time(void * self, double * time);
        """
        pass

    def get_end_time(self):
        """End time of the model.

        Returns
        -------
        float
          The maximum model time.

        See Also
        --------
        get_start_time

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_end_time(void * self, double * time);
        """
        pass

    def get_time_step(self):
        """Current time step of the model.

        The model time step should be of type float. The default time
        step is 1.0.

        Returns
        -------
        float
          The time step used in model.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_time_step(void * self, double * dt);
        """
        pass

    def get_time_units(self):
        """Time units of the model.

        CSDMS uses the UDUNITS standard from Unidata.

        Returns
        -------
        float
          The model time unit; e.g., `days` or `s`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_time_units(void * self, char * units);
        """
        pass
