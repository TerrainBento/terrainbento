# coding: utf8
# !/usr/env/python

import pytest

import numpy as np

from terrainbento.output_writers import StaticIntervalOutputWriter

# Helper classes and functions
class ClockModel:
    def __init__(self, clock):
        self.clock = clock
def to_floats(int_list):
    return [None if i is None else float(i) for i in int_list]

# Fixtures
@pytest.fixture()
def clock08_model(clock_08):
    return ClockModel(clock_08)


''' #test_names (delete?)
@pytest.mark.parametrize("in_name, add_id, out_name", [
    (None, True, "static-interval-output-writer"),
    (None, False, "static-interval-output-writer"),
    ('given_nameT', True, "given_nameT"),
    ('given_nameF', False, "given_nameF"),
    ])
def test_names(clock08_model, in_name, add_id, out_name):
    # Make a few writer to check if the id value is handled correctly
    for i in range(3):
        if in_name is None:
            writer = StaticIntervalOutputWriter(
                    clock08_model,
                    add_id=add_id,
                    )
        else:
            writer = StaticIntervalOutputWriter(
                    clock08_model,
                    name=in_name,
                    add_id=add_id,
                    )
        if add_id:
            assert writer.name == out_name + f"-id{writer.id}"
        else:
            assert writer.name == out_name
            
'''
def test_not_implemented_functions(clock08_model):
    writer = StaticIntervalOutputWriter(clock08_model)
    base_functions = [
            writer.run_one_step,
            ]
    for fu in base_functions:
        with pytest.raises(NotImplementedError):
            fu()


@pytest.mark.parametrize("intervals, times, error_type", [
    ([1,2,3], [1,2,3], AssertionError), # Both args defined
    ('a', None, NotImplementedError), # Bad arg type
    (None, 'a', NotImplementedError), # Bad arg type
    (['a'], None, NotImplementedError), # Bad arg type in list
    (None, ['a'], NotImplementedError), # Bad arg type in list
    ])
def test_interval_times_bad_input(clock08_model, intervals, times, error_type):
    """
    Test that the correct errors are thrown for bad input.
    """
    with pytest.raises(error_type):
        writer = StaticIntervalOutputWriter(
                clock08_model,
                intervals=intervals, 
                times=times, 
                )

@pytest.mark.parametrize("intervals, times, output_times", [
    # None at the end of the output list enforces that the iterator exhausts

    # Test default and scalar behavior 
    (None, None, [20, None]), # stop time for clock_08
    (5  , None, [5, 10, 15, 20, None]), # Test single interval duration (int)
    (5.0, None, [5, 10, 15, 20, None]), # Test single interval duration (float)
    (None, 5  , [5, None]), # Test single output time (int)
    (None, 5.0, [5, None]), # Test single output time (float)

    # Test lists of floats and ints
    ([1  ,2  ,3  ], None, [1, 3, 6, None]), # Test list of integer intervals
    ([1.0,2.0,3.0], None, [1, 3, 6, None]), # Test list of float intervals
    (None, [1  ,2  ,3  ], [1, 2, 3, None]), # Test list of integer times
    (None, [1.0,2.0,3.0], [1, 2, 3, None]), # Test list of float times
    ])
def test_intervals_times_correct_input(clock08_model, intervals, times, output_times):
    """
    Test all the different variations of correct input formats.
    """
    writer = StaticIntervalOutputWriter(
            clock08_model,
            intervals=intervals, 
            times=times, 
            )

    for correct_out in iter(to_floats(output_times)):
        writer_out = writer.advance_iter()
        print(writer_out, correct_out)

        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float 
            assert writer_out == correct_out

def test_intervals_repeat(clock08_model):
    """
    Test if a repeating list of intervals will produce the right output times.
    """
    intervals = [1, 2, 3]
    output_times = [1.0, 3.0, 6.0, 7.0, 9.0, 12.0, 13.0, 15.0, 18.0, 19.0, None]
    # Note that clock_08 stop time is 20.0
    # None at the end of the output list enforces that the iterator exhausts

    writer = StaticIntervalOutputWriter(
            clock08_model,
            intervals=intervals, 
            intervals_repeat=True,
            times=None, 
            )

    for correct_out in output_times:
        writer_out = writer.advance_iter()
        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float 
            assert writer_out == correct_out
    
    # Note that the writer.times_iter will be an infinite iterator.

@pytest.mark.parametrize("intervals, times, output_times", [
    # None at the end of the output list enforces that the iterator exhausts

    # Test default and scalar behavior 
    (None, None, [0, 20, None]), # stop time for clock_08
    (5  , None, [0, 5, 10, 15, 20, None]), # Test single interval duration (int)
    (5.0, None, [0, 5, 10, 15, 20, None]), # Test single interval duration (float)
    (None, 5  , [0, 5, None]), # Test single output time (int)
    (None, 5.0, [0, 5, None]), # Test single output time (float)

    # Test lists of floats and ints that don't include a time zero
    ([1  ,2  ,3  ], None, [0, 1, 3, 6, None]), # Test list of integer intervals
    ([1.0,2.0,3.0], None, [0, 1, 3, 6, None]), # Test list of float intervals
    (None, [1  ,2  ,3  ], [0, 1, 2, 3, None]), # Test list of integer times
    (None, [1.0,2.0,3.0], [0, 1, 2, 3, None]), # Test list of float times

    # Test lists of floats and ints that already include a time zero
    ([0, 1  ,2  ,3  ], None, [0, 1, 3, 6, None]), # Test list of integer intervals
    ([0, 1.0,2.0,3.0], None, [0, 1, 3, 6, None]), # Test list of float intervals
    (None, [0, 1  ,2  ,3  ], [0, 1, 2, 3, None]), # Test list of integer times
    (None, [0, 1.0,2.0,3.0], [0, 1, 2, 3, None]), # Test list of float times
    ])
def test_correct_input_with_firststep(clock08_model, intervals, times, output_times):
    """
    Test that needing output on the first step still works.
    """
    writer = StaticIntervalOutputWriter(
            clock08_model,
            intervals=intervals, 
            times=times, 
            save_first_timestep=True,
            )
    for correct_out in to_floats(output_times):
        writer_out = writer.advance_iter()
        if correct_out is None:
            assert writer_out is None 
        else:
            assert type(writer_out) is float 
            assert writer_out == correct_out
