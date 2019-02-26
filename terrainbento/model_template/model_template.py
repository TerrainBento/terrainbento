"""A template for making a derived terrainbento model.

This template shows all of the required parts of a new terrainbento model,
designed and created by **you**

In this part of the documentation, we make sure to include hyperlinks to all
landlab components used.
"""

# import any major python libraries needed
# e.g. sys, os, numpy, scipy, pandas

# import all required parts of landlab
# e.g. landlab components you want to use

from terrainbento.base_class import ErosionModel


class ModelTemplate(ErosionModel):  # The model must inherit from either
    # ErosionModel, StochasticErosionModel, or TwoLithologyErosionModel
    """ModelTemplate is a template for making your own terrainbento models.

    This is where you will put introductory information about the model. We
    recommend that you start from an existing terrainbento model"s docstring
    and modify to preserve a somewhat standard style.

    The docstring should have:

    1. A brief description of the model.

    2. Links to all landlab components used.

    3. Description of the governing equation of the model.
    """

    _required_fields = ["topographic__elevation"]

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodibility=0.0001,
        regolith_transport_parameter=0.1,
        **kwargs
    ):
        """
        Parameters
        ----------


        Examples
        --------
        This is where you can make code examples showing how to use the model
        you created. Here we typically put a very short example that shows a
        minimally complete parameter dictionary for creating an instance of the
        model.

        Then in unit tests we include all possible analytical solutions and
        assertion tests needed to verify the model program is working as
        expected.

        *For example*: This is a minimal example to demonstrate how to
        construct an instance of model **ModelTemplate**. Note that a YAML
        input file can be used instead of a parameter dictionary. For more
        detailed examples, including steady-state test examples, see the
        terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento.model_template import ModelTemplate

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicStVs
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = ModelTemplate(clock, grid)

        """
        # Replace  `ModelTemplate` with your model name.
        super(ModelTemplate, self).__init__(clock, grid, **kwargs)
        # Do not change any additional parts of this line.

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # put all actions needed to initialize the model below this line.

    def run_one_step(self, step):
        """Run each component for one time step.

        Put any additional information about **run_one_step** here.
        Importantly, **run_one_step** should only take on parameter,
        ``step``.
        """
        # write here all actions needed to run the model forward for a time
        # increment `step`.

        # end with finalize__run_one_step which does things at the end of
        # run_one_step that are required by all models.
        self.finalize__run_one_step(step)

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

    model = ModelTemplate.from_file(infile)
    model.run()


if __name__ == "__main__":
    main()
