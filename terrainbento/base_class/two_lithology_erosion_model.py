# coding: utf8
#! /usr/env/python
"""
Base class for common functions of terrainbentostochastic erosion models.

The **TwoLithologyErosionModel** is a base class that contains all of the
functionality shared by the terrainbento models that have two lithologies.
"""

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
        self.contact_width = (self._length_factor) * self.params[
            "contact_zone__width"
        ]  # has units length

        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

    def _setup_contact_elevation(self):
        file_name = self.params["lithology_contact_elevation__file_name"]

        # Read input data on rock-till contact elevation
        read_esri_ascii(
            file_name, grid=self.grid, halo=1, name="rock_till_contact__elevation"
        )
        self.rock_till_contact = self.grid.at_node["rock_till_contact__elevation"]

    def _update_erodywt(self):
        # Update the erodability weighting function (this is "F")
        self.erody_wt[self.data_nodes] = 1.0 / (
            1.0
            + np.exp(
                -(self.z[self.data_nodes] - self.rock_till_contact[self.data_nodes])
                / self.contact_width
            )
        )
