# coding: utf8
# !/usr/env/python

import pytest

from landlab.components import (
    DepressionFinderAndRouter,
    FlowDirectorMFD,
    FlowDirectorSteepest,
)
from terrainbento import ErosionModel
from terrainbento.utilities import *


def test_FlowAccumulator_with_depression_steepest():
    params = {
        "model_grid": "RasterModelGrid",
        "clock": SIMPLE_CLOCK,
        "flow_director": "FlowDirectorSteepest",
        "depression_finder": "DepressionFinderAndRouter",
    }

    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorSteepest)
    assert isinstance(
        em.flow_accumulator.depression_finder, DepressionFinderAndRouter
    )


def test_no_depression_finder():
    params = {"model_grid": "RasterModelGrid", "clock": SIMPLE_CLOCK}

    em = ErosionModel(params=params)
    assert em.flow_accumulator.depression_finder is None


def test_FlowAccumulator_with_D8_Hex():
    params = {
        "model_grid": "HexModelGrid",
        "clock": SIMPLE_CLOCK,
        "flow_director": "D8",
    }
    pytest.raises(NotImplementedError, ErosionModel, params=params)


def test_FlowAccumulator_with_depression_MFD():
    params = {
        "model_grid": "HexModelGrid",
        "clock": SIMPLE_CLOCK,
        "flow_director": "MFD",
    }
    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorMFD)


def test_alt_names_steepest():
    params = {
        "model_grid": "RasterModelGrid",
        "clock": SIMPLE_CLOCK,
        "flow_director": "D4",
    }

    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorSteepest)

    params = {
        "model_grid": "RasterModelGrid",
        "clock": SIMPLE_CLOCK,
        "flow_director": "Steepest",
    }

    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorSteepest)
