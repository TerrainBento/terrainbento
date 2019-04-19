#! /usr/bin/env python
"""Interface for getting and setting a model's internal variables."""


class BmiGetter(object):

    """Get values from a component.

    Methods that get variables from a model's state. Often a model's state
    variables are changing with each time step, so getters are called to get
    current values.
    """

    def get_value(self, var_name):
        """Get a copy of values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a *copy* of a model variable, with
        the return type, size and rank dependent on the variable.

        Parameters
        ----------
        var_name : str
          An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        array_like
          The value of a model variable.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_value(void * self, const char * var_name, void * buffer);
        """
        pass

    def get_value_ref(self, var_name):
        """Get a reference to values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a reference to a model variable,
        with the return type, size and rank dependent on the variable.

        Parameters
        ----------
        var_name : str
          An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        array_like
          A reference to a model variable.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_value_ref(void * self, const char * var_name,
                              void ** buffer);
        """
        pass

    def get_value_at_indices(self, var_name, indices):
        """Get values at particular indices.

        Parameters
        ----------
        var_name : str
          An input or output variable name, a CSDMS Standard Name.
        indices : array_like
          The indices into the variable array.

        Returns
        -------
        array_like
            Value of the model variable at the given location.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_value_at_indices(void * self, const char * var_name,
                                     void * buffer, int * indices, int len);
        """
        pass


class BmiSetter(object):

    """Set values into a component.

    Methods that set variables of a model's state.
    """

    def set_value(self, var_name, src):
        """Specify a new value for a model variable.

        This is the setter for the model, used to change the model's
        current state. It accepts, through *src*, a new value for a
        model variable, with the type, size and rank of *src*
        dependent on the variable.

        Parameters
        ----------
        var_name : str
          An input or output variable name, a CSDMS Standard Name.
        src : array_like
          The new value for the specified variable.

        Notes
        -----
        .. code-block:: c

            /* C */
            int set_value(void * self, const char * var_name, void * src);
        """
        pass

    def set_value_at_indices(self, var_name, indices, src):
        """Specify a new value for a model variable at particular indices.

        Parameters
        ----------
        var_name : str
          An input or output variable name, a CSDMS Standard Name.
        indices : array_like
          The indices into the variable array.
        src : array_like
          The new value for the specified variable.

        Notes
        -----
        .. code-block:: c

            /* C */
            int set_value_at_indices(void * self, const char * var_name,
                                     int * indices, int len, void * src);
        """
        pass
