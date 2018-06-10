# -*- coding: utf-8 -*-
"""
``terrainbento`` Model 000 Basic

Erosion model program using linear diffusion, stream power, and discharge
proportional to drainage area.

Landlab components used: FlowAccumulator, DepressionFinderAndRouter,
                         FastscapeStreamPower, LinearDiffuser
"""
import sys
import numpy as np

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class Basic(ErosionModel):
    """Model 000 Basic program.

    Model 000 Basic is a model program that evolves a topographic surface desribed
    by :math:`\eta` with the following governing equation:

    .. math::

        \frac{\partial \eta}{\partial t} = - KA^{M_{sp}}S^{n_sp} + D\nabla^2 \eta

    where :math:`A` is the local drainage area and :math:`S` is the local slope.

    Model 000 Basic inherits from the ``terrainbento`` ``ErosionModel`` base
    class. Depending on the values of :math:`K`, :math:`D`, :math:`m_{sp}` and,
    :math:`n_{sp}` this model program can be used to run the following three
    ``terrainbento`` numerical models.

    1) Model 000 Basic: Here :math:`m_{sp}` has a value of 0.5 and :math:`n_{sp}`
    has a value of 1. :math:`K` is given by the parameter ``water_erodability``
    and :math:`D` is given by the parameter ``regolith_transport_parameter``.

    2) Model 001 BasicVm: This model is identical to Model 000 Basic except that
    the area exponent :math:`m_{sp}` is a free parameter.

    3) Model 004 BasicSs: In this model :math:`m_{sp}` has a value of 1/3,
    :math:`n_{sp}` has a value of 2/3, and :math:`K` is given by the paramter
    :math:`water_erodability~shear_stress`.

    In addition to those provided by the ``ErosionModel`` base class  ``Basic``
    has the following attributes and methods.

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params is
            required.
        BoundaryHandlers : class or list of classes, optional
            Classes used to handle boundary conditions. Alternatively can be
            passed by input file as string. Valid options described above.
        OutputWriters : class, function, or list of classes and/or functions, optional
        Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).

        Returns
        -------
        ErosionModel : object

        Examples
        --------
        We recommend that you look at the ``terraintento`` tutorials for
        examples of usage.
        """
        # Call ErosionModel's init
        super(Basic, self).__init__(input_file=input_file,
                                    params=params,
                                    BoundaryHandlers=BoundaryHandlers,
                                    OutputWriters=OutputWriters)

        # Get Parameters:
        K_sp = self.get_parameter_from_exponent('water_erodability', raise_error=False)
        K_ss = self.get_parameter_from_exponent('water_erodability~shear_stress', raise_error=False)
        regolith_transport_parameter = (self._length_factor**2.) * self.get_parameter_from_exponent('regolith_transport_parameter') # has units length^2/time

        # check that a stream power and a shear stress parameter have not both been given
        if K_sp != None and K_ss != None:
            raise ValueError(('Model 000: A parameter for both '
                              'water_erodability and '
                              'water_erodability~shear_stress has been provided. '
                              ' Only one of these may be provided.'))
        elif K_sp != None or K_ss != None:
            if K_sp != None:
                self.K = K_sp
            else:
                self.K = (self._length_factor**(1./3.))*K_ss # K_ss has units Length^(1/3) per Time
        else:
            raise ValueError(('water_erodability or '
                              'water_erodability~shear_stress must be provided.'))

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid,
                                      K_sp=self.K,
                                      m_sp=self.params['m_sp'],
                                      n_sp=self.params['n_sp'])

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = regolith_transport_parameter)

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        # Route flow
        self.flow_accumulator.run_one_step()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(self.flow_accumulator.depression_finder.flood_status==3)[0]

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if 'PrecipChanger' in self.boundary_handler:
            self.eroder.K = (self.K
                             * self.boundary_handler['PrecipChanger'].get_erodibility_adjustment_factor())
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print(('To run a terrainbento model from the command line you must '
                'include input file name on command line'))
        sys.exit(1)

    model = Basic(input_file=infile)
    model.run()


if __name__ == '__main__':
    main()
