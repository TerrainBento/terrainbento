import pytest

from landlab import RasterModelGrid
from terrainbento import Clock


@pytest.fixture()
def U():
    U = 0.0001
    return U


@pytest.fixture()
def K():
    K = 0.01
    return K


@pytest.fixture()
def Kr():
    Kr = 0.01
    return Kr


@pytest.fixture()
def Kt():
    Kt = 0.02
    return Kt


@pytest.fixture()
def grid_1():
    grid = RasterModelGrid((3, 21), xy_spacing=100.0)
    grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
    grid.add_zeros("node", "topographic__elevation")
    grid.add_ones("node", "soil__depth")
    grid.add_zeros("node", "lithology_contact__elevation")
    return grid


@pytest.fixture()
def grid_2():
    grid = RasterModelGrid((8, 20), xy_spacing=100.0)
    grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
    grid.add_zeros("node", "topographic__elevation")
    grid.add_ones("node", "soil__depth")
    lith = grid.add_zeros("node", "lithology_contact__elevation")
    lith[:80] = 10
    lith[80:] = -10000.0
    return grid


@pytest.fixture()
def grid_3():
    grid = RasterModelGrid((21, 3), xy_spacing=100.0)
    grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
    grid.add_zeros("node", "topographic__elevation")
    grid.add_ones("node", "soil__depth")
    lith = grid.add_zeros("node", "lithology_contact__elevation")
    lith[grid.core_nodes[:9]] = -100000.0
    lith[grid.core_nodes[9:]] = 100000.0
    return grid


@pytest.fixture()
def grid_4():
    grid = RasterModelGrid((3, 21), xy_spacing=10.0)
    grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
    grid.add_zeros("node", "topographic__elevation")
    grid.add_ones("node", "soil__depth")
    lith = grid.add_zeros("node", "lithology_contact__elevation")
    lith[grid.core_nodes[:9]] = -100000.0
    lith[grid.core_nodes[9:]] = 100000.0
    return grid


@pytest.fixture
def grid_5():
    grid = RasterModelGrid((6, 9), xy_spacing=10)
    grid.add_zeros("node", "topographic__elevation")
    grid.add_ones("node", "soil__depth")
    lith = grid.add_zeros("node", "lithology_contact__elevation")
    lith[:27] = -30
    lith[27:] = 10.0
    lith[grid.boundary_nodes] = -9999.0
    return grid


@pytest.fixture
def almost_default_grid():
    grid = RasterModelGrid((4, 5), xy_spacing=100.0)
    grid.add_zeros("node", "topographic__elevation")
    return grid


@pytest.fixture
def simple_square_grid():
    grid = RasterModelGrid((10, 10), xy_spacing=10)
    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
    grid.add_zeros("node", "topographic__elevation")
    return grid


@pytest.fixture()
def clock_simple():
    clock_simple = Clock(step=1000.0, stop=5.1e6)
    return clock_simple


@pytest.fixture()
def clock_02():
    clock_02 = Clock.from_dict({"step": 10.0, "stop": 1000.0})
    return clock_02


@pytest.fixture()
def clock_04():
    clock_04 = Clock.from_dict({"step": 10.0, "stop": 100000.0})
    return clock_04


@pytest.fixture()
def clock_05():
    clock_05 = Clock.from_dict({"step": 10.0, "stop": 200.0})
    return clock_05


@pytest.fixture()
def clock_06():
    clock_06 = Clock.from_dict({"step": 1.0, "stop": 3.0})
    return clock_06


@pytest.fixture()
def clock_07():
    clock_07 = Clock.from_dict({"step": 10.0, "stop": 10000.0})
    return clock_07


@pytest.fixture()
def clock_08():
    clock_08 = Clock(step=1.0, stop=20.0)
    return clock_08


@pytest.fixture()
def clock_09():
    clock_09 = Clock(step=2.0, stop=200.0)
    return clock_09


@pytest.fixture()
def precip_defaults():
    precip_defaults = {
        "daily_rainfall__intermittency_factor": 0.5,
        "daily_rainfall__intermittency_factor_time_rate_of_change": 0.1,
        "rainfall__mean_rate": 1.0,
        "rainfall__mean_rate_time_rate_of_change": 0.2,
        "infiltration_capacity": 0,
        "rainfall__shape_factor": 0.65,
    }
    return precip_defaults


@pytest.fixture()
def precip_testing_factor():
    precip_testing_factor = 1.3145341380253433
    return precip_testing_factor


@pytest.fixture()
def clock_yaml():
    out = """
    start: 1
    step: 2
    stop: 11
    """
    return out


@pytest.fixture()
def inputs_yaml():
    out = """
    grid:
      HexModelGrid:
        - base_num_rows: 8
          base_num_cols: 5
          dx: 10
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    clock:
      step: 1
      stop: 10.

    output_interval: 2.
    """
    return out


@pytest.fixture()
def bad_handler_yaml():
    out = """
    grid:
      RasterModelGrid:
        - [4, 5]
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    clock:
      step: 100
      stop: 2000.

    boundary_handlers:
      spam_and_eggs:
        foo: 100
        bar: -0.01
    """
    return out


@pytest.fixture()
def basic_inputs_bad_precipitator_yaml():
    out = """
    grid:
      HexModelGrid:
        - base_num_rows: 8
          base_num_cols: 5
          dx: 10
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    clock:
      step: 1
      stop: 200.

    output_interval: 50.
    water_erodibility: 0.001
    m_sp: 1
    n_sp: 0.5
    regolith_transport_parameter: 0.01

    precipitator:
      UniformPrecipitator:
        rainfall_flux: 3
      SomethingElse:
        anotherkwarg: 100.
    """
    return out


@pytest.fixture()
def basic_inputs_no_clock_yaml():
    out = """
    grid:
      HexModelGrid:
        - base_num_rows: 8
          base_num_cols: 5
          dx: 10
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    output_interval: 50.
    water_erodibility: 0.001
    m_sp: 1
    n_sp: 0.5
    regolith_transport_parameter: 0.01
    """
    return out


@pytest.fixture()
def basic_inputs_no_grid_yaml():
    out = """
    clock:
      step: 1
      stop: 200.

    output_interval: 50.
    water_erodibility: 0.001
    m_sp: 1
    n_sp: 0.5
    regolith_transport_parameter: 0.01
    """
    return out


@pytest.fixture()
def basic_inputs_yaml():
    out = """
    grid:
      HexModelGrid:
        - base_num_rows: 8
          base_num_cols: 5
          dx: 10
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    clock:
      step: 1
      stop: 200.

    output_interval: 50.
    water_erodibility: 0.001
    m_sp: 1
    n_sp: 0.5
    regolith_transport_parameter: 0.01
    """
    return out


@pytest.fixture()
def inputs_D8_yaml():
    out = """
    grid:
      RasterModelGrid:
        - [4, 5]
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    clock:
      step: 1
      stop: 10.

    output_interval: 2.
    flow_director: "FlowDirectorD8"
    """
    return out


@pytest.fixture()
def basic_raster_inputs_yaml():
    out = """
    grid:
      RasterModelGrid:
        - [4, 5]
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    clock:
      step: 1
      stop: 200.
    water_erodibility: 0.001
    m_sp: 1
    n_sp: 0.5
    regolith_transport_parameter: 0.01
    output_interval: 50
    """
    return out


@pytest.fixture()
def basic_raster_inputs_for_nc_yaml():
    out = """
    grid:
      RasterModelGrid:
        - [4, 5]
        - fields:
            node:
              topographic__elevation:
                constant:
                  - value: 0.0
    clock:
      step: 100
      stop: 2000.
    water_erodibility: 0.0001
    m_sp: 1
    n_sp: 0.5
    regolith_transport_parameter: 0.01
    output_interval: 500

    boundary_handlers:
      NotCoreNodeBaselevelHandler:
        modify_core_nodes: True
        lowering_rate: -0.01
    fields:
      - topographic__elevation
      - cumulative_elevation_change
      - initial_topographic__elevation
      - water__unit_flux_in
      - flow__sink_flag
      - flow__receiver_node
      - topographic__steepest_slope
      - flow__link_to_receiver_node
      - drainage_area
      - surface_water__discharge
      - flow__upstream_node_order
      - flow__data_structure_delta
    """
    return out
