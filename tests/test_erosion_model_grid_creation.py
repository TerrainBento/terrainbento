# coding: utf8
#! /usr/env/python

import os
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from landlab import HexModelGrid, RasterModelGrid
from terrainbento import ErosionModel


_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

at_node_fields = [
    "topographic__elevation",
    "initial_topographic__elevation",
    "cumulative_erosion__depth",
    "water__unit_flux_in",
    "flow__receiver_node",
    "topographic__steepest_slope",
    "flow__link_to_receiver_node",
    "flow__sink_flag",
    "drainage_area",
    "surface_water__discharge",
    "flow__upstream_node_order",
    "flow__data_structure_delta",
]


def test_HexModelGrid_default():
    params = {
        "model_grid": "HexModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)
    assert isinstance(em.grid, HexModelGrid)
    assert em.grid.number_of_nodes == 56
    for field in at_node_fields:
        assert field in em.grid.at_node


def test_RasterModelGrid_default():
    params = {"dt": 1, "output_interval": 2., "run_duration": 10.}
    em = ErosionModel(params=params)
    assert isinstance(em.grid, RasterModelGrid)
    assert em.grid.number_of_nodes == 20
    assert em.grid.number_of_node_columns == 5
    assert em.grid.number_of_node_rows == 4
    assert em.grid.dx == 1.0
    for field in at_node_fields:
        assert field in em.grid.at_node


def test_default_sythetic_topo():
    params = {"dt": 1, "output_interval": 2., "run_duration": 10.}
    em = ErosionModel(params=params)
    assert np.array_equiv(em.z, 0.0) == True


def test_no_noise_sythetic_topo():
    params = {
        "initial_elevation": 10.,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
        "add_noise_to_all_nodes": True,
    }
    em = ErosionModel(params=params)
    known_z = np.zeros(em.z.shape)
    known_z += 10.
    assert np.array_equiv(em.z, known_z) == True


def test_noise_all_nodes_sythetic_topo():
    params = {
        "initial_elevation": 10.,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
        "add_random_noise": True,
        "add_noise_to_all_nodes": True,
        "initial_noise_std": 2.0,
    }
    em = ErosionModel(params=params)
    known_z = np.zeros(em.z.shape)
    np.random.seed(0)
    rs = np.random.randn(em.grid.number_of_nodes)
    known_z += 10. + (2. * rs)
    assert_array_equal(known_z, em.z)

    np.random.seed(0)
    rs = np.random.randn(len(em.grid.core_nodes))


def test_sythetic_topo_default_seed():
    params = {
        "add_random_noise": True,
        "initial_elevation": 10.,
        "initial_noise_std": 2.,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)

    known_z = np.zeros(em.z.shape)
    np.random.seed(0)
    rs = np.random.randn(len(em.grid.core_nodes))
    known_z[em.grid.core_nodes] += 10. + (2. * rs)
    assert_array_equal(known_z, em.z)


def test_sythetic_topo_set_seed():
    params = {
        "add_random_noise": True,
        "initial_elevation": 10.,
        "initial_noise_std": 2.,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
        "random_seed": 42,
    }
    em = ErosionModel(params=params)
    known_z = np.zeros(em.z.shape)
    np.random.seed(42)
    rs = np.random.randn(len(em.grid.core_nodes))
    known_z[em.grid.core_nodes] += 10. + (2. * rs)
    assert_array_equal(known_z, em.z)


def test_Hex_with_outlet():
    params = {
        "model_grid": "HexModelGrid",
        "number_of_node_rows": 5,
        "number_of_node_columns": 4,
        "node_spacing": 10,
        "outlet_id": 9,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)
    assert em.outlet_node == 9
    assert em.opt_watershed == True
    status = np.array(
        [4, 4, 4, 4, 4, 0, 0, 0, 4, 1, 0, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4]
    )
    assert_array_equal(status, em.grid.status_at_node)


def test_Hex_with_outlet_not_specified():
    params = {
        "model_grid": "HexModelGrid",
        "number_of_node_rows": 5,
        "number_of_node_columns": 4,
        "node_spacing": 10,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)
    assert em.outlet_node == 0
    assert em.opt_watershed == False
    status = np.array(
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
    )
    assert_array_equal(status, em.grid.status_at_node)


def test_Hex_with_boundaries():
    params = {
        "model_grid": "HexModelGrid",
        "number_of_node_rows": 5,
        "number_of_node_columns": 4,
        "node_spacing": 10,
        "boundary_closed": True,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)
    assert em.outlet_node == 0
    assert em.opt_watershed == False
    status = np.array(
        [4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4]
    )
    assert_array_equal(status, em.grid.status_at_node)


def test_Raster_with_outlet():
    params = {
        "model_grid": "RasterModelGrid",
        "number_of_node_rows": 5,
        "number_of_node_columns": 4,
        "node_spacing": 10,
        "outlet_id": 3,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)
    assert em.outlet_node == 3
    assert em.opt_watershed == True
    status = np.array([4, 4, 4, 1, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 4, 4])
    assert_array_equal(status, em.grid.status_at_node)


def test_Raster_with_outlet_not_specified():
    params = {
        "model_grid": "RasterModelGrid",
        "number_of_node_rows": 5,
        "number_of_node_columns": 4,
        "node_spacing": 10,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)
    assert em.outlet_node == 0
    assert em.opt_watershed == False
    status = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1])
    assert_array_equal(status, em.grid.status_at_node)


def test_Raster_with_boundaries():
    params = {
        "model_grid": "RasterModelGrid",
        "number_of_node_rows": 5,
        "number_of_node_columns": 4,
        "east_boundary_closed": True,
        "west_boundary_closed": True,
        "node_spacing": 10,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }
    em = ErosionModel(params=params)
    assert em.outlet_node == 0
    assert em.opt_watershed == False
    status = np.array([1, 1, 1, 1, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 1, 1, 1, 1])
    assert_array_equal(status, em.grid.status_at_node)


def test_DEM_ascii():
    fp = os.path.join(_TEST_DATA_DIR, "test_4_x_3.asc")
    params = {"DEM_filename": fp, "dt": 1, "output_interval": 2., "run_duration": 10.}

    em = ErosionModel(params=params)

    assert isinstance(em.grid, RasterModelGrid)

    assert isinstance(em.z, np.ndarray)
    core_vals = em.z.reshape(em.grid.shape)[1:-1, 1:-1].flatten()
    assert_array_equal(
        core_vals, np.array([9., 10., 11., 6., 7., 8., 3., 4., 5., 0., 1., 2.])
    )
    assert em.outlet_node == 21
    assert em.z[em.outlet_node] == 0.0
    assert em.opt_watershed == True


def test_DEM_two_possible_outlets():
    fp = os.path.join(_TEST_DATA_DIR, "test_4_x_3_two_zeros.asc")
    params = {"DEM_filename": fp, "dt": 1, "output_interval": 2., "run_duration": 10.}

    pytest.raises(ValueError, ErosionModel, params=params)

    fp = os.path.join(_TEST_DATA_DIR, "test_4_x_3_two_zeros.asc")
    params = {
        "DEM_filename": fp,
        "outlet_id": 22,
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 10.,
    }

    em = ErosionModel(params=params)
    assert em.outlet_node == 22


def test_DEM_netcdf():
    """Test DEM."""
    fp = os.path.join(_TEST_DATA_DIR, "test_file.nc")
    params = {"DEM_filename": fp, "dt": 1, "output_interval": 2., "run_duration": 10.}

    em = ErosionModel(params=params)

    assert isinstance(em.grid, RasterModelGrid)

    assert isinstance(em.z, np.ndarray)
    core_vals = em.z.reshape(em.grid.shape)[1:-1, 1:-1].flatten()
    assert_array_equal(
        core_vals, np.array([9., 10., 11., 6., 7., 8., 3., 4., 5., 0., 1., 2.])
    )
    assert em.outlet_node == 21
    assert em.z[em.outlet_node] == 0.0
    assert em.opt_watershed == True
