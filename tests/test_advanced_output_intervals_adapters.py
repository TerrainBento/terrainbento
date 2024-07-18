# !/usr/env/python

import pytest

from terrainbento.output_writers import (
    StaticIntervalOutputClassAdapter,
    StaticIntervalOutputFunctionAdapter,
)


# Helper functions and classes
def to_floats(int_list):
    return [None if i is None else float(i) for i in int_list]


class BasicModel:
    def __init__(self, clock, step):
        self.clock = clock
        self.step = step
        self.model_time = 0.0


# Simple output writer class
class SimpleOutputClass:
    def __init__(self, model):
        self.model = model
        self.record = []

    def run_one_step(self):
        self.record.append(self.model.model_time)


# Simple output writer function
FUNCTION_RECORD = []


def simple_output_function(model):
    FUNCTION_RECORD.append(model.model_time)


# Fixtures
@pytest.fixture()
def basic_model(clock_08):
    return BasicModel(
        clock_08,
    )


def test_class_adapter(clock_08):
    """
    Test that the class adapter initializes and produces the right output
    times. Not sure how else I can test the code.
    """

    step = 1.0
    basic_model = BasicModel(clock_08, step)

    output_interval = 6
    correct_outputs = to_floats([0, 6, 12, 18, 20])
    # Note: clock_08 stops at 20 and save_last_timestep = True

    writer = StaticIntervalOutputClassAdapter(
        basic_model,
        output_interval,
        SimpleOutputClass,
        save_first_timestep=True,
        save_last_timestep=True,
    )

    # Run "model"
    stop_time = basic_model.clock.stop
    writer.advance_iter()
    while basic_model.model_time <= stop_time:
        if basic_model.model_time == writer.next_output_time:
            writer.run_one_step()
            writer.advance_iter()
        basic_model.model_time += basic_model.step

    # Check output times
    ow_record = writer.ow_class.record
    assert len(ow_record) == len(correct_outputs)
    for correct_out, writer_out in zip(correct_outputs, ow_record):
        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float
            assert writer_out == correct_out


def test_function_adapter(clock_08):
    """
    Test that the function adapter initializes and produces the right output
    times. Not sure how else I can test the code.
    """

    step = 1.0
    basic_model = BasicModel(clock_08, step)

    output_interval = 3
    correct_outputs = to_floats([0, 3, 6, 9, 12, 15, 18, 20])
    # Note: clock_08 stops at 20 and save_last_timestep = True

    writer = StaticIntervalOutputFunctionAdapter(
        basic_model,
        output_interval,
        simple_output_function,
        save_first_timestep=True,
        save_last_timestep=True,
    )

    # Run "model"
    stop_time = basic_model.clock.stop
    writer.advance_iter()
    while basic_model.model_time <= stop_time:
        print(f"Model time: {basic_model.model_time}")
        if basic_model.model_time == writer.next_output_time:
            print("  Output!")
            writer.run_one_step()
            writer.advance_iter()
        basic_model.model_time += basic_model.step

    # Check output times
    ow_record = FUNCTION_RECORD
    print("Writer:  ", ow_record)
    print("Correct: ", correct_outputs)
    assert len(ow_record) == len(correct_outputs)
    for correct_out, writer_out in zip(correct_outputs, ow_record):
        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float
            assert writer_out == correct_out
