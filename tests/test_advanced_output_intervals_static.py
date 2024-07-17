# !/usr/env/python

import pytest

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


""" #test_names (delete?)
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

"""


def test_not_implemented_functions(clock08_model):
    writer = StaticIntervalOutputWriter(clock08_model)
    base_functions = [
        writer.run_one_step,
    ]
    for fu in base_functions:
        with pytest.raises(NotImplementedError):
            fu()


@pytest.mark.parametrize(
    "intervals, times, error_type",
    [
        ([1, 2, 3], [1, 2, 3], AssertionError),  # Both args defined
        ("a", None, NotImplementedError),  # Bad arg type
        (None, "a", NotImplementedError),  # Bad arg type
        (["a"], None, NotImplementedError),  # Bad arg type in list
        (None, ["a"], NotImplementedError),  # Bad arg type in list
        (0, None, AssertionError),  # Interval of zero makes no sense
        ([0, 2, 3], None, AssertionError),  # Interval of zero makes no sense
    ],
)
def test_bad_input(clock08_model, intervals, times, error_type):
    """
    Test that the correct errors are thrown for bad input.
    """
    with pytest.raises(error_type):
        _ = StaticIntervalOutputWriter(
            clock08_model,
            intervals=intervals,
            times=times,
        )


@pytest.mark.parametrize(
    "intervals, times, output_times",
    [
        # None at the end of the output list enforces that the iterator exhausts
        # Test default and scalar behavior
        (None, None, [20, None]),  # stop time for clock_08
        (
            5,
            None,
            [5, 10, 15, 20, None],
        ),  # Test single interval duration (int)
        (
            5.0,
            None,
            [5, 10, 15, 20, None],
        ),  # Test single interval duration (float)
        (None, 5, [5, None]),  # Test single output time (int)
        (None, 5.0, [5, None]),  # Test single output time (float)
        # Test lists of floats and ints
        ([1, 2, 3], None, [1, 3, 6, None]),  # Test list of integer intervals
        (
            [1.0, 2.0, 3.0],
            None,
            [1, 3, 6, None],
        ),  # Test list of float intervals
        (None, [1, 2, 3], [1, 2, 3, None]),  # Test list of integer times
        (None, [1.0, 2.0, 3.0], [1, 2, 3, None]),  # Test list of float times
    ],
)
def test_correct_input_plain(clock08_model, intervals, times, output_times):
    """
    Test all the different variations of correct input formats.
    """
    writer = StaticIntervalOutputWriter(
        clock08_model,
        intervals=intervals,
        intervals_repeat=False,
        times=times,
        save_first_timestep=False,
        save_last_timestep=False,
    )

    for correct_out in iter(to_floats(output_times)):
        writer_out = writer.advance_iter()
        print(writer_out, correct_out)

        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float
            assert writer_out == correct_out


@pytest.mark.parametrize(
    "intervals, times, output_times",
    [
        # None at the end of the output list enforces that the iterator exhausts
        # Test default and scalar behavior
        (None, None, [0, 20, None]),  # stop time for clock_08
        (5, None, [0, 5, 10, 15, 20, None]),  # Single interval duration (int)
        (
            5.0,
            None,
            [0, 5, 10, 15, 20, None],
        ),  # Single interval duration (float)
        (None, 5, [0, 5, None]),  # Single output time (int)
        (None, 5.0, [0, 5, None]),  # Single output time (float)
        # Test lists of floats and ints that don't include a time zero
        ([1, 2, 3], None, [0, 1, 3, 6, None]),  # List of integer intervals
        ([1.0, 2.0, 3.0], None, [0, 1, 3, 6, None]),  # List of float intervals
        (None, [1, 2, 3], [0, 1, 2, 3, None]),  # List of integer times
        (None, [1.0, 2.0, 3.0], [0, 1, 2, 3, None]),  # List of float times
        # Test lists of floats and ints that already include a time zero
        (None, [0, 1, 2, 3], [0, 1, 2, 3, None]),  # List of integer times
        (None, [0, 1.0, 2.0, 3.0], [0, 1, 2, 3, None]),  # List of float times
    ],
)
def test_correct_input_with_firststep(clock08_model, intervals, times, output_times):
    """
    Test that needing output on the first step works.
    """
    writer = StaticIntervalOutputWriter(
        clock08_model,
        intervals=intervals,
        intervals_repeat=False,
        times=times,
        save_first_timestep=True,
        save_last_timestep=False,
    )
    for correct_out in to_floats(output_times):
        writer_out = writer.advance_iter()
        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float
            assert writer_out == correct_out


@pytest.mark.parametrize(
    "intervals, times, output_times",
    [
        # None at the end of the output list enforces that the iterator exhausts
        # Test default and scalar behavior
        (None, None, [20, None]),  # stop time for clock_08
        (5, None, [5, 10, 15, 20, None]),  # Single interval duration (int)
        (5.0, None, [5, 10, 15, 20, None]),  # Single interval duration (float)
        (None, 5, [5, 20, None]),  # Single output time (int)
        (None, 5.0, [5, 20, None]),  # Single output time (float)
        # Test lists of floats and ints that don't include the stop time
        ([1, 2, 3], None, [1, 3, 6, 20, None]),  # List of integer intervals
        (
            [1.0, 2.0, 3.0],
            None,
            [1, 3, 6, 20, None],
        ),  # List of float intervals
        (None, [1, 2, 3], [1, 2, 3, 20, None]),  # List of integer times
        (None, [1.0, 2.0, 3.0], [1, 2, 3, 20, None]),  # List of float times
        # Test lists of floats and ints that already include the stop time
        (None, [1, 2, 3, 20], [1, 2, 3, 20, None]),  # List of integer times
        (
            None,
            [1.0, 2.0, 3.0, 20.0],
            [1, 2, 3, 20, None],
        ),  # List of float times
        # Test intervals that jump the stop time
        (6, None, [6, 12, 18, 20, None]),  # Single interval jumps stop time
        (
            [1, 2, 3, 50],
            None,
            [1, 3, 6, 20, None],
        ),  # List intervals jumps stop time
    ],
)
def test_correct_input_with_laststep(clock08_model, intervals, times, output_times):
    """
    Test that needing output on the last step works.
    """
    writer = StaticIntervalOutputWriter(
        clock08_model,
        intervals=intervals,
        intervals_repeat=False,
        times=times,
        save_first_timestep=False,
        save_last_timestep=True,
    )
    for correct_out in to_floats(output_times):
        writer_out = writer.advance_iter()
        print(writer_out, correct_out)
        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float
            assert writer_out == correct_out


@pytest.mark.parametrize(
    "save_last",
    [
        (False),
        (True),
    ],
)
def test_repeating_intervals_list(clock08_model, save_last):
    """
    Test if a repeating list of intervals will produce the right output times.
    """
    intervals = [1, 2, 3]
    output_times_ints = [1, 3, 6, 7, 9, 12, 13, 15, 18, 19]
    output_times_ints += [20, None] if save_last else [None]
    output_times = to_floats(output_times_ints)
    # Note that clock_08 stop time is 20.0
    # None at the end of the output list enforces that the iterator exhausts

    writer = StaticIntervalOutputWriter(
        clock08_model,
        intervals=intervals,
        intervals_repeat=True,
        times=None,
        save_first_timestep=False,
        save_last_timestep=save_last,
    )

    for correct_out in output_times:
        writer_out = writer.advance_iter()
        if correct_out is None:
            assert writer_out is None
        else:
            assert type(writer_out) is float
            assert writer_out == correct_out

    # Note that the writer.times_iter will be an infinite iterator.
