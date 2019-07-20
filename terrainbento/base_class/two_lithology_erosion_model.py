# coding: utf8
# !/usr/env/python
"""Base class for common functions of terrainbento models with two lithologies.

The **TwoLithologyErosionModel** is a base class that contains all of
the functionality shared by the terrainbento models that have two
lithologies.
"""
import numpy as np

from terrainbento.base_class import ErosionModel


class TwoLithologyErosionModel(ErosionModel):
    """Base class for two lithology terrainbento models.

    A **TwoLithologyErosionModel** inherits from
    :py:class:`ErosionModel` and
    provides functionality needed by all models with two lithologies.

    This is a base class that handles setting up common parameters and the
    contact zone elevation.

    The following at-node fields must be specified in the grid:
        - ``topographic__elevation``
        - ``lithology_contact__elevation``
    """

    _required_fields = [
        "topographic__elevation",
        "lithology_contact__elevation",
    ]

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodibility_lower=0.0001,
        water_erodibility_upper=0.001,
        regolith_transport_parameter=0.1,
        contact_zone__width=1.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        clock : terrainbento Clock instance
        grid : landlab model grid instance
            The grid must have all required fields.
        **kwargs :
            Keyword arguments to pass to :py:class:`ErosionModel`. Importantly
            these arguments specify the precipitator and the runoff generator
            that control the generation of surface water discharge (:math:`Q`).

        Returns
        -------
        TwoLithologyErosionModel : object

        Examples
        --------
        This model is a base class and is not designed to be run on its own. We
        recommend that you look at the terrainbento tutorials for examples of
        usage.
        """
        # Call ErosionModel"s init
        super(TwoLithologyErosionModel, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        self.m = m_sp
        self.n = n_sp

        # Get all common parameters
        self.contact_width = contact_zone__width

        self.regolith_transport_parameter = regolith_transport_parameter

        self.K_rock = water_erodibility_lower

        self.K_till = water_erodibility_upper

        # Set the erodibility values, these need to be double stated because a PrecipChanger may adjust them
        self.rock_erody = self.K_rock
        self.till_erody = self.K_till

        self.rock_till_contact = self.grid.at_node[
            "lithology_contact__elevation"
        ]

    def _setup_rock_and_till(self):
        """Set up fields to handle for two layers with different
        erodibility."""

        # Create field for erodibility
        self.erody = self.grid.add_zeros("node", "substrate__erodibility")

        # Create array for erodibility weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

        # Set values correctly
        self._update_erodywt()
        self._update_erodibility_field()

    def _setup_rock_and_till_with_threshold(self):
        """Set up fields to handle for two layers with different
        erodibility."""

        # Create field for erodibility
        self.erody = self.grid.add_zeros("node", "substrate__erodibility")

        # Create field for threshold values
        self.threshold = self.grid.add_zeros(
            "node", "water_erosion_rule__threshold"
        )

        # Create array for erodibility weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

        # set values correctly
        self._update_erodywt()
        self._update_erodibility_and_threshold_fields()

    def _update_erodywt(self):
        # Update the erodibility weighting function (this is "F")
        core = self.grid.core_nodes
        if self.contact_width > 0.0:
            self.erody_wt[core] = 1.0 / (
                1.0
                + np.exp(
                    -(self.z[core] - self.rock_till_contact[core])
                    / self.contact_width
                )
            )
        else:
            self.erody_wt[core] = 0.0
            self.erody_wt[np.where(self.z > self.rock_till_contact)[0]] = 1.0

    def _update_Ks_with_precip(self):
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handlers:
            erode_factor = self.boundary_handlers[
                "PrecipChanger"
            ].get_erodibility_adjustment_factor()
            self.till_erody = self.K_till * erode_factor
            self.rock_erody = self.K_rock * erode_factor

    def _update_erodibility_field(self):
        """Update erodibility at each node.

        The erodibility at each node is a smooth function between the
        rock and till erodabilities and is based on the contact zone
        width and the elevation of the surface relative to contact
        elevation.
        """
        self._update_erodywt()
        self._update_Ks_with_precip()

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (
            self.erody_wt * self.till_erody
            + (1.0 - self.erody_wt) * self.rock_erody
        )

    def _update_erodibility_and_threshold_fields(self):
        """Update erodibility at each node.

        The erodibility at each node is a smooth function between the
        rock and till erodabilities and is based on the contact zone
        width and the elevation of the surface relative to contact
        elevation.
        """
        self._update_erodywt()
        self._update_Ks_with_precip()

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (
            self.erody_wt * self.till_erody
            + (1.0 - self.erody_wt) * self.rock_erody
        )

        # Calculate the effective thresholds using weighted averaging
        self.threshold[:] = (
            self.erody_wt * self.till_thresh
            + (1.0 - self.erody_wt) * self.rock_thresh
        )
