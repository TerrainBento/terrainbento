#! /usr/env/python
"""
model_010_basicHy.py: calculates water erosion using the
hybrid alluvium model.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         Space

"""
import sys
import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicHy(ErosionModel):
    """
    A BasicHy model computes erosion of sediment and bedrock
    using dual mass conservation on the bed and in the water column. It
    applies exponential entrainment rules to account for bed cover.
    """

    def __init__(self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None):
        """Initialize the HybridAlluviumModel."""

        # Call ErosionModel's init
        super(BasicHy, self).__init__(input_file=input_file,
                                      params=params,
                                      BoundaryHandlers=BoundaryHandlers,
                                        OutputWriters=OutputWriters)

        # Get Parameters and convert units if necessary:
        K_sp = self.get_parameter_from_exponent('water_erodability', raise_error=False)
        K_ss = self.get_parameter_from_exponent('water_erodability~shear_stress', raise_error=False)

        # check that a stream power and a shear stress parameter have not both been given
        if K_sp != None and K_ss != None:
            raise ValueError('A parameter for both K_rock_sp and K_rock_ss has been'
                             'provided. Only one of these may be provided')
        elif K_sp != None or K_ss != None:
            if K_sp != None:
                self.K = K_sp
            else:
                self.K = (self._length_factor**(1./3.))*K_ss # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError('A value for K_rock_sp or K_rock_ss  must be provided.')

        # Unit conversion for linear_diffusivity, with units L^2/T
        regolith_transport_parameter = ((self._length_factor ** 2.)
            * self.get_parameter_from_exponent('regolith_transport_parameter'))

        # Normalized settling velocity (dimensionless)
        v_sc = self.get_parameter_from_exponent('v_sc')

        #make area_field and/or discharge_field depending on discharge_method
#        area_field = self.grid.at_node['drainage_area']
#        discharge_field = None

        # Handle solver option
        try:
            solver = self.params['solver']
        except:
            solver = 'original'

        # Instantiate a Space component
        self.eroder = ErosionDeposition(self.grid,
                            K=self.K,
                            phi=self.params['phi'],
                            F_f=self.params['F_f'],
                            v_s=v_sc,
                            m_sp=self.params['m_sp'],
                            n_sp=self.params['n_sp'],
                            method='simple_stream_power',
                            discharge_method='drainage_area',
                            area_field='drainage_area',
                            solver=solver)

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
        self.eroder.run_one_step(dt,
                                 flooded_nodes=flooded,
                                 dynamic_dt=True,
                                 flow_director=self.flow_accumulator.flow_director)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main():
    """Execute model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    ha = BasicHy(input_file=infile)
    ha.run()


if __name__ == '__main__':
    main()
