"""A template for making a derived terrainbento model.

This template shows all of the required parts of a new terrainbento model,
designed and created by **you**
"""

# import any major python libraries needed
# e.g. sys, os, numpy, scipy, pandas

# import all required parts of landlab
# e.g. landlab components you want to use

from terrainbento.base_class import ErosionModel


class ModelTemplate(ErosionModel):  # The model must inherit from either
    # ErosionModel or StochasticErosionModel
    """ModelTemplate is a template for making your own terrainbento models.

    This is where you will put introductory information about the model.

    You should probably define the governing equation of your model here.

    The **ModelTemplate** program inherits from the terrainbento
    **ErosionModel** base class. In addition to the parameters required by
    the base class, models built with this program require the following parameters.

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Parameter Name        |
    +==================+==================================+
    |:math:`m`         | ``m_sp``                         |
    +------------------+----------------------------------+
    |:math:`n`         | ``n_sp``                         |
    +------------------+----------------------------------+
    |:math:`K`         | ``water_erodability``            |
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+

    Expand on this table to include all required parameters.

    """

    def __init__(self, input_file=None, params=None, OutputWriters=None):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params is
            required.
        OutputWriters : class, function, or list of classes and/or functions, optional
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).

        Examples
        --------
        This is where you can make code examples showing how to use the model you
        created.

        >>> from terrainbento.model_template import ModelTemplate
        >>> # this is where you'd show how to import and use your model.
        >>> # these statements get evaluated in testing so its also a way to show
        >>> # that the model does what you say it will do.
        >>> # its important to make sure that all lines of your model are tested
        >>> # either in these docstring tests or in test files.

        """
        super(ModelTemplate, self).__init__(
            input_file=input_file,  # Replace  `ModelTemplate` with your model name.
            params=params,  # Do not change any additional parts of this
        )  # line.

        # put all actions needed to initialize the model below this line.

    def run_one_step(self, dt):
        """Run each component for one time step.

        Put any additional information about **run_one_step** here. Importantly,
        **run_one_step** should only take on parameter, ``dt``.
        """
        # write here all actions needed to run the model forward for a time
        # increment `dt`.

        # end with finalize__run_one_step which does things at the end of
        # run_one_step that are required by all models.
        self.finalize__run_one_step(dt)

    # if you have additional class functions, you can define them here.
    def my_internal_function(self):
        """Do something necessary to instantiate or run ``ModelTemplate``."""
        # replace pass with function.
        pass

    # if your model has required finalization steps, define them here. This
    # definition will overwrite the empty `finalize` definition in the
    # `ErosionModel`.

    def finalize(self):
        """Finalize model.

        Put additional information about finalizing the model here.
        """
        # replace pass with all actions needed to finalize the model.
        # if you are inheriting from the stochastic erosion model, be careful
        # here as it already has a **finalize** method defined.
        pass


# this portion makes it possible to run the model from the command line.
def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print(
            (
                "To run a terrainbento model from the command line you must "
                "include input file name on command line"
            )
        )
        sys.exit(1)

    model = ModelTemplate(input_file=infile)
    model.run()


if __name__ == "__main__":
    main()
