# coding: utf8
#! /usr/env/python
"""
Base class for common functions of terrainbentostochastic erosion models.

The **TwoLithologyErosionModel** is a base class that contains all of the
functionality shared by the terrainbento models that have two lithologies.
"""
import numpy as np

from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel


class TwoLithologyErosionModel(ErosionModel):
    """Base class for two lithology terrainbento models.

    A **TwoLithologyErosionModel** inherits from **ErosionModel** and provides
    functionality needed by all models with two lithologies.

    This is a base class that handles setting up common parameters and the
    contact zone elevation.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params
            is required.
        BoundaryHandlers : class or list of classes, optional
            Classes used to handle boundary conditions. Alternatively can be
            passed by input file as string. Valid options described above.
        OutputWriters : class, function, or list of classes and/or functions,
            optional classes or functions used to write incremental output
            (e.g. make a diagnostic plot).

        Returns
        -------
        TwoLithologyErosionModel : object

        Examples
        --------
        This model is a base class and is not designed to be run on its own. We
        recommend that you look at the terrainbento tutorials for examples of
        usage.
        """
        # Call ErosionModel's init
        super(TwoLithologyErosionModel, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get all common parameters
        self.contact_width = (self._length_factor) * self.params[
            "contact_zone__width"
        ]  # has units length

        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]

        self.regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        self.K_rock = self.get_parameter_from_exponent("water_erodability~lower") * (
            self._length_factor ** (1. - (2. * self.m))
        )

        self.K_till = self.get_parameter_from_exponent("water_erodability~upper") * (
            self._length_factor ** (1. - (2. * self.m))
        )

        # Set the erodability values, these need to be double stated because a PrecipChanger may adjust them
        self.rock_erody = self.K_rock
        self.till_erody = self.K_till

    def _setup_contact_elevation(self):
        file_name = self.params["lithology_contact_elevation__file_name"]

        # Read input data on rock-till contact elevation
        read_esri_ascii(
            file_name, grid=self.grid, halo=1, name="rock_till_contact__elevation"
        )
        self.rock_till_contact = self.grid.at_node["rock_till_contact__elevation"]

    def _setup_rock_and_till(self):
        """Set up fields to handle for two layers with different erodability."""
        # Get a reference to the rock-till field
        self._setup_contact_elevation()

        # Create field for erodability
        self.erody = self.grid.add_zeros("node", "substrate__erodability")

        # Create array for erodability weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

        # Set values correctly
        self._update_erodywt()
        self._update_erodability_field()

    def _setup_rock_and_till_with_threshold(self):
        """Set up fields to handle for two layers with different erodability."""
        # Get a reference to the rock-till field\
        self._setup_contact_elevation()

        # Create field for erodability
        self.erody = self.grid.add_zeros("node", "substrate__erodability")

        # Create field for threshold values
        self.threshold = self.grid.add_zeros("node", "water_erosion_rule__threshold")

        # Create array for erodability weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

        # set values correctly
        self._update_erodywt()
        self._update_erodability_and_threshold_fields()

    def _update_erodywt(self):
        # Update the erodability weighting function (this is "F")
        core = self.grid.core_nodes
        if self.contact_width > 0.0:
            self.erody_wt[core] = 1.0 / (
                1.0
                + np.exp(
                    -(self.z[core] - self.rock_till_contact[core]) / self.contact_width
                )
            )
        else:
            self.erody_wt[core] = 0.0
            self.erody_wt[np.where(self.z > self.rock_till_contact)[0]] = 1.0

    def _update_Ks_with_precip(self):
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            erode_factor = self.boundary_handler[
                "PrecipChanger"
            ].get_erodability_adjustment_factor()
            self.till_erody = self.K_till * erode_factor
            self.rock_erody = self.K_rock * erode_factor

    def _update_erodability_field(self):
        """Update erodability at each node.

        The erodability at each node is a smooth function between the rock and
        till erodabilities and is based on the contact zone width and the
        elevation of the surface relative to contact elevation.
        """
        self._update_erodywt()
        self._update_Ks_with_precip()

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (
            self.erody_wt * self.till_erody + (1.0 - self.erody_wt) * self.rock_erody
        )

    def _update_erodability_and_threshold_fields(self):
        """Update erodability at each node.

        The erodability at each node is a smooth function between the rock and
        till erodabilities and is based on the contact zone width and the
        elevation of the surface relative to contact elevation.
        """
        self._update_erodywt()
        self._update_Ks_with_precip()

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (
            self.erody_wt * self.till_erody + (1.0 - self.erody_wt) * self.rock_erody
        )

        # Calculate the effective thresholds using weighted averaging
        self.threshold[:] = (
            self.erody_wt * self.till_thresh + (1.0 - self.erody_wt) * self.rock_thresh
        )
