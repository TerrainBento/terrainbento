#! /usr/bin/env python

STANDARD_NAME = {
    "drainage_area": "basin__total_contributing_area",
    "flow__link_to_receiver_node": "model_grid_node_link~downstream__index",
    "flow__receiver_node": None,
    "flow__sink_flag": None,  # model_*__flag boolean?
    "flow__upstream_node_order": None,
    "topographic__elevation": "land_surface__elevation",
    "topographic__gradient": "land_surface__slope",
    "topographic__slope": "land_surface__slope_angle",
    "topographic__steepest_slope": "model_grid_cell__max_of_d8_slope",
    "soil__depth": "spam",
    "lithology_contact__elevation": "eggs",
    "surface_water__discharge": "land_surface_water__volume_flow_rate",
    "water__unit_flux_in": "model_grid_cell_water~incoming__volume_flow_rate",
}


TERRAINBENTO_NAME = dict(
    (value, key) for key, value in STANDARD_NAME.items() if key
)
