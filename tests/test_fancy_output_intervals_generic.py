# coding: utf8
# !/usr/env/python

import pytest
from .conftest import clock_08

import itertools
from terrainbento.output_writers import (
        GenericOutputWriter, 
        OutputIteratorSkipWarning,
        )


# Helper classes and functions
class EmptyModel:
    def __init__(self):
        self.clock = clock_08
def to_floats(int_list):
    return [None if i is None else float(i) for i in int_list]


# Test basic properties and attributes
def test_id():
    """ Test that the id generator is working correctly. """
    class OutputA (GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-A")

    class OutputB (GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-B")

    class OutputC (GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-C")

    empty_model = EmptyModel()
    correct_id = itertools.count()
    for i in range(25):
        for cls_type in OutputA, OutputB, OutputC, GenericOutputWriter:
            # should make 100 output classes in total (4 types x 25)
            writer = cls_type(empty_model)
            assert writer.id == next(correct_id)

@pytest.mark.parametrize("in_name, add_id, out_name", [
    (None, True, "output-writer"),
    (None, False, "output-writer"),
    ('given_nameT', True, "given_nameT"),
    ('given_nameF', False, "given_nameF"),
    ])
def test_names(in_name, add_id, out_name):
    empty_model = EmptyModel()
    for i in range(3):
        writer = GenericOutputWriter(
                empty_model,
                name=in_name,
                add_id=add_id,
                )
        if add_id:
            assert writer.name == out_name + f"-id{writer.id}"
        else:
            assert writer.name == out_name

def test_not_implemented():
    empty_model = EmptyModel()
    writer = GenericOutputWriter(empty_model)
    base_functions = [
            writer.run_one_step,
            ]
    for fu in base_functions:
        with pytest.raises(NotImplementedError):
            fu()

# Test the iterator behaviour
@pytest.mark.parametrize("times_iter, error_type", [
    (None, AssertionError),          # _times_iter is None
    ([1,2,3], AssertionError),       # _times_iter doesn't have next
    (iter([1,2,3]), AssertionError), # _times_iter doesn't returns floats
    ])
def test_times_iter_bad_input(times_iter, error_type):
    """ Test that errors while advancing the iterator are called correctly. """
    empty_model = EmptyModel()
    writer = GenericOutputWriter(empty_model)
    writer.register_times_iter(times_iter)

    with pytest.raises(error_type):
        writer.advance_iter()

def test_times_iter_max_recursion():
    """ Test that warnings are raised for skipping values and an error is 
    thrown for too many skips."""
    times_iter = iter(to_floats([6, 5,4,3,2,1,0]))
    empty_model = EmptyModel()
    writer = GenericOutputWriter(empty_model)
    writer.register_times_iter(times_iter)

    # Advance to the first item successfully
    writer.advance_iter()

    # Advance to the second item, which causes the recursion chain
    with pytest.warns(OutputIteratorSkipWarning), pytest.raises(RecursionError):
        # Note: order of 'with' statement matters. Failures from 'warns' won't 
        # pass through 'raises' correctly if order reversed.
        writer.advance_iter()


@pytest.mark.parametrize("times_ints, save_first, output_ints", [
    ([1,2,3],   False, [1,2,3, None]), # Normal
    ([1,2,3],   True,  [0,1,2,3, None]), # save first step; no skips needed
    ([0,1,2,3], False, [0,1,2,3, None]), # Normal
    ([1,2,3],   False, [1,2,3, None, None, None]), # exhausted iterator
    ([1,2,3],   True,  [0,1,2,3, None, None, None]), # exhausted iterator
    ])
def test_times_iter_correct_no_skips(times_ints, save_first, output_ints):
    """ Test that the times iterator can be added and advanced correctly. """
    times_iter = iter(to_floats(times_ints))
    output_iter = iter(to_floats(output_ints))

    empty_model = EmptyModel()
    writer = GenericOutputWriter(empty_model, save_first_timestep=save_first)
    writer.register_times_iter(times_iter)

    correct_previous = None
    for correct_out in output_iter:
        writer_out = writer.advance_iter()
        assert writer_out is writer.next_output_time

        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float 
            assert writer_out == correct_out
            
        writer_previous = writer.prev_output_time
        if correct_previous is None:
            assert writer_previous is None
        else:
            assert correct_previous == writer_previous
        # Update correct_previous var but ignore if iterator is exhausted
        if correct_out is not None:
            correct_previous = correct_out

@pytest.mark.parametrize("times_ints, save_first, n_skip_warnings, output_ints", [
    # Poorly made sequence:
    ([0,1,2,0,1,2,3], False, [0,0,0,3,0], [0,1,2,3,None]),
    # Multiple skipping areas (that each need less the 5 skips):
    ([1,1,1,2,2,2,3,3,3], False, [0,2,2,2], [1,2,3,None]),
    # save first step; skip extra zero but NO warning (only for zeros)
    ([0,1,2,3], True, [0,0,0,0,0], [0,1,2,3,None]),
    ([0,0,0,1,2,3], True, [0,0,0,0,0], [0,1,2,3,None]),
    ])
def test_times_iter_correct_with_skips(
        times_ints, 
        save_first, 
        n_skip_warnings, 
        output_ints
        ):
    """ Test that the times iterator can be added and advanced correctly 
    despite needing to skip a few values. """

    # In case you add a new test scenario incorrectly...
    assert len(n_skip_warnings) == len(output_ints), ''.join([
            f"Bad testing setup: n_skip_warnings (len = ", 
            f"{len(n_skip_warnings)}) needs to be the same length as ", 
            f"output_ints (len = {len(output_ints)})",
            ])

    # Convert all the input lists into iterators
    times_iter = iter(to_floats(times_ints))
    skip_iter = iter(n_skip_warnings)
    output_iter = iter(to_floats(output_ints))

    # Set up the output writer
    empty_model = EmptyModel()
    writer = GenericOutputWriter(empty_model, save_first_timestep=save_first)
    writer.register_times_iter(times_iter)

    # Loop through all the correct outputs to see if the writer generates them.
    correct_previous = None
    for correct_out, n_skip in zip(output_iter, skip_iter):
        if n_skip == 0:
            # No skip warnings should be produced. (warnings fail the test)
            writer_out = writer.advance_iter()
        else:
            # n_skip warnings are expected
            with pytest.warns(OutputIteratorSkipWarning) as warning_list:
                writer_out = writer.advance_iter()
            assert n_skip == len(warning_list)

        if correct_out is None:
            # The iterator is exhausted
            assert writer_out is None
        else:
            # The iterator generated a value. Make sure it is a float and 
            # matches the correct output.
            assert type(writer_out) is float 
            assert writer_out == correct_out
            
        # Check that the previous output is correct despite skips or exhaustion
        writer_previous = writer.prev_output_time
        if correct_previous is None:
            assert writer_previous is None
        else:
            assert correct_previous == writer_previous

        # Update correct_previous var but ignore if iterator is exhausted
        if correct_out is not None:
            correct_previous = correct_out
