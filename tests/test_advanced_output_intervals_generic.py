# !/usr/env/python

import itertools
import os.path

import pytest

from terrainbento.output_writers import GenericOutputWriter, OutputIteratorSkipWarning


# Helper classes and functions
class ClockModel:
    def __init__(self, clock):
        self.clock = clock


def to_floats(int_list):
    return [None if i is None else float(i) for i in int_list]


def generate_previous_list(next_list):
    """
    Generate the expected list of previous values given a list of next values.
    Lags the next list, holds the last valid number, and adds None to the
    front. e.g.
    [0,1,2,None,None,None] -> [None, 0,1,2,2,2]
    """
    # Add none and lag the list
    previous_list = [None] + next_list[:-1]

    # Hold the last valid number by replacing None with last valid number
    idx_last_valid = 1
    for i in range(1, len(previous_list)):
        if previous_list[i] is None:
            previous_list[i] = previous_list[idx_last_valid]
        else:
            idx_last_valid = i

    assert len(next_list) == len(previous_list)
    return previous_list


# Fixtures
@pytest.fixture()
def clock08_model(clock_08):
    return ClockModel(clock_08)


# Test basic properties and attributes
def test_id(clock08_model):
    """Test that the id generator is working correctly."""

    class OutputA(GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-A")

    class OutputB(GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-B")

    class OutputC(GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-C")

    # Other output writer tests may occur before this one. So the ID generator
    # might not start at zero
    start_id = next(GenericOutputWriter._id_iter)
    correct_id = itertools.count(start=start_id + 1)
    for i in range(25):
        for cls_type in OutputA, OutputB, OutputC, GenericOutputWriter:
            # should make 100 output classes in total (4 types x 25)
            writer = cls_type(clock08_model)
            assert writer.id == next(correct_id)


@pytest.mark.parametrize(
    "in_name, add_id, out_name",
    [
        (None, True, "output-writer"),
        (None, False, "output-writer"),
        ("given_nameT", True, "given_nameT"),
        ("given_nameF", False, "given_nameF"),
    ],
)
def test_names(clock08_model, in_name, add_id, out_name):
    for i in range(3):
        writer = GenericOutputWriter(
            clock08_model,
            name=in_name,
            add_id=add_id,
        )
        if add_id:
            assert writer.name == out_name + f"-id{writer.id}"
        else:
            assert writer.name == out_name


def test_not_implemented_functions(clock08_model):
    writer = GenericOutputWriter(clock08_model)
    base_functions = [
        writer.run_one_step,
    ]
    for fu in base_functions:
        with pytest.raises(NotImplementedError):
            fu()


def test_clock_required():
    """
    Test that errors are thrown if the model is missing a clock.
    """

    class EmptyModel:
        pass

    with pytest.raises(AssertionError):
        _ = GenericOutputWriter(EmptyModel())

    no_clock_model = ClockModel(None)
    with pytest.raises(AssertionError):
        _ = GenericOutputWriter(no_clock_model)


# Test the iterator behaviour
@pytest.mark.parametrize(
    "times_iter, error_type",
    [
        (None, AssertionError),  # _times_iter is None
        ([1, 2, 3], AssertionError),  # _times_iter doesn't have next
        (
            iter([1, 2, 3]),
            AssertionError,
        ),  # _times_iter doesn't returns floats
    ],
)
def test_times_iter_bad_iterator(clock08_model, times_iter, error_type):
    """Test that the correct errors are thrown for bad iterators."""
    writer = GenericOutputWriter(clock08_model)
    writer.register_times_iter(times_iter)

    with pytest.raises(error_type):
        writer.advance_iter()


def test_times_iter_max_recursion(clock08_model):
    """Test that warnings are raised for skipping values and an error is
    thrown for too many skips."""
    times_iter = iter(to_floats([6, 5, 4, 3, 2, 1, 0]))
    writer = GenericOutputWriter(clock08_model)
    writer.register_times_iter(times_iter)

    # Advance to the first item successfully
    writer.advance_iter()

    # Advance to the second item, which triggers the recursion chain
    with pytest.warns(OutputIteratorSkipWarning), pytest.raises(RecursionError):
        # Note: order of 'with' statement matters. Failures from 'warns' won't
        # pass through 'raises' correctly if order reversed.
        writer.advance_iter()


@pytest.mark.parametrize(
    "times_ints, save_first, save_last, output_ints",
    [
        # 4 binary input differences (has 0, has 20, save_first, save_last) means
        # 16 test. There is a lot of overlap.
        #
        # Normal input
        # - -  F F
        # 0 -  F F
        # - 20 F F
        # 0 20 F F
        ([1, 2, 3], False, False, [1, 2, 3, None]),  # Normal
        ([0, 1, 2, 3], False, False, [0, 1, 2, 3, None]),  # Normal
        ([1, 2, 3, 20], False, False, [1, 2, 3, 20, None]),  # Normal
        ([0, 1, 2, 3, 20], False, False, [0, 1, 2, 3, 20, None]),  # Normal
        #
        # No skips needed for enforcing initial or final times
        # - -  F T
        # - -  T F
        # - -  T T
        ([1, 2, 3], False, True, [1, 2, 3, 20, None]),  # Save last step
        ([1, 2, 3], True, False, [0, 1, 2, 3, None]),  # Save first step
        (
            [1, 2, 3],
            True,
            True,
            [0, 1, 2, 3, 20, None],
        ),  # Save first and last steps
        # 0 -  F T
        # - 20 T F
        ([0, 1, 2, 3], False, True, [0, 1, 2, 3, 20, None]),  # Save last step
        (
            [1, 2, 3, 20],
            True,
            False,
            [0, 1, 2, 3, 20, None],
        ),  # Save first step
        #
        # Skips needed for enforcing initial and/or final times, but there should
        # be no warning
        # 0 -  T F
        # 0 -  T T
        ([0, 1, 2, 3], True, False, [0, 1, 2, 3, None]),  # Save first step
        (
            [0, 1, 2, 3],
            True,
            True,
            [0, 1, 2, 3, 20, None],
        ),  # Save first and last steps
        # - 20 F T
        # - 20 T T
        ([1, 2, 3, 20], False, True, [1, 2, 3, 20, None]),  # Save last step
        (
            [1, 2, 3, 20],
            True,
            True,
            [0, 1, 2, 3, 20, None],
        ),  # Save first and last steps
        # 0 20 F T
        # 0 20 T F
        # 0 20 T T
        (
            [0, 1, 2, 3, 20],
            False,
            True,
            [0, 1, 2, 3, 20, None],
        ),  # Save last step
        (
            [0, 1, 2, 3, 20],
            True,
            False,
            [0, 1, 2, 3, 20, None],
        ),  # Save first step
        (
            [0, 1, 2, 3, 20],
            True,
            True,
            [0, 1, 2, 3, 20, None],
        ),  # Save first and last steps
        # Explicitly test iterator exhaustion
        # Don't bother with all the permutations bc that is above.
        ([1, 2, 3], False, False, [1, 2, 3, None, None, None]),  # Normal
        (
            [0, 1, 2, 3, 20],
            True,
            True,
            [0, 1, 2, 3, 20, None, None, None],
        ),  # Save first and last steps
        # Tests iterators that jump past stop time
        ([1, 2, 3, 40], False, False, [1, 2, 3, None, None, None]),  # Normal
        (
            [1, 2, 3, 40],
            True,
            True,
            [0, 1, 2, 3, 20, None, None, None],
        ),  # Enforce saving last step
        # Test that iterators exhaust if any value goes over stop time
        ([1, 2, 3, 40, 15], False, False, [1, 2, 3, None, None, None]),
        ([1, 2, 3, 40, 15], True, True, [0, 1, 2, 3, 20, None, None, None]),
        # Test multiple zero or stop time skips
        (
            [0, 0, 0, 0, 0, 1, 2, 3],
            True,
            False,
            [0, 1, 2, 3, None],
        ),  # Save first step
        (
            [1, 2, 3, 20, 20, 20, 20],
            False,
            True,
            [1, 2, 3, 20, None],
        ),  # Save last step
    ],
)
def test_times_iter_correct_no_skip_warnings(
    clock08_model, times_ints, save_first, save_last, output_ints
):
    """Test that the times iterator can be added and advanced correctly."""
    times_iter = iter(to_floats(times_ints))
    output_times = to_floats(output_ints)
    previous_times = generate_previous_list(output_times)

    writer = GenericOutputWriter(
        clock08_model,
        save_first_timestep=save_first,
        save_last_timestep=save_last,
    )
    writer.register_times_iter(times_iter)

    for correct_out, correct_previous in zip(output_times, previous_times):
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


@pytest.mark.parametrize(
    "times_ints, save_first, n_skip_warnings, output_ints",
    [
        # Poorly made sequence:
        ([0, 1, 2, 0, 1, 2, 3], False, [0, 0, 0, 3, 0], [0, 1, 2, 3, None]),
        # Multiple skipping areas (that each need less the 5 skips):
        ([1, 1, 1, 2, 2, 2, 3, 3, 3], False, [0, 2, 2, 2], [1, 2, 3, None]),
    ],
)
def test_times_iter_correct_with_skip_warnings(
    clock08_model, times_ints, save_first, n_skip_warnings, output_ints
):
    """Test that the times iterator can be added and advanced correctly
    despite needing to skip a few values."""

    # In case you add a new test scenario incorrectly...
    assert len(n_skip_warnings) == len(output_ints), "".join(
        [
            "Bad testing setup: n_skip_warnings (len = ",
            f"{len(n_skip_warnings)}) needs to be the same length as ",
            f"output_ints (len = {len(output_ints)})",
        ]
    )

    # Setup up inputs and outputs if necessary
    times_iter = iter(to_floats(times_ints))
    output_times = to_floats(output_ints)
    previous_times = generate_previous_list(output_times)

    # Set up the output writer
    writer = GenericOutputWriter(
        clock08_model,
        save_first_timestep=save_first,
        save_last_timestep=False,
    )
    writer.register_times_iter(times_iter)

    # Loop through all the correct outputs to see if the writer generates them.
    for correct_out, n_skip, correct_previous in zip(
        output_times, n_skip_warnings, previous_times
    ):
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


def test_times_iter_infinite_iter(clock08_model):
    """
    Test that an infinite iterator can be added and advanced correctly.
    """
    start = 0.0
    step = 10.0
    times_iter = (start + step * i for i in itertools.count())

    # Note: clock_08 stops at 20.0
    output_times = to_floats([0, 10, 20, None, None, None])
    previous_times = generate_previous_list(output_times)

    writer = GenericOutputWriter(clock08_model)
    writer.register_times_iter(times_iter)

    for correct_out, correct_previous in zip(output_times, previous_times):
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


# Test Files
def test_output_files(clock08_model):
    """Test that the file tracking functions are working correctly."""

    class OutputA(GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-A", add_id=False)

    class OutputB(GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-B", add_id=False)

    class OutputC(GenericOutputWriter):
        def __init__(self, model):
            super().__init__(model, name="class-C", add_id=False)

    output_a = OutputA(clock08_model)
    output_b = OutputB(clock08_model)
    output_c = OutputC(clock08_model)
    output_g = GenericOutputWriter(clock08_model, add_id=False)

    output_dir = output_a.output_dir

    correct_files_dict = {
        output_a: [
            os.path.join(output_dir, "class-A_file-0.txt"),
            os.path.join(output_dir, "class-A_file-1.txt"),
            os.path.join(output_dir, "class-A_file-2.md"),
        ],
        output_b: [
            os.path.join(output_dir, "class-B_file-0.txt"),
            os.path.join(output_dir, "class-B_file-1.txt"),
            os.path.join(output_dir, "class-B_file-2.md"),
        ],
        output_c: [
            os.path.join(output_dir, "class-C_file-0.txt"),
            os.path.join(output_dir, "class-C_file-1.txt"),
            os.path.join(output_dir, "class-C_file-2.md"),
        ],
        output_g: [
            os.path.join(output_dir, "generic_file-0.txt"),
            os.path.join(output_dir, "generic_file-1.txt"),
            os.path.join(output_dir, "generic_file-2.md"),
        ],
    }

    for writer, filepaths in correct_files_dict.items():
        for filepath in filepaths:
            open(filepath, "w").close()
            writer.register_output_filepath(filepath)

            assert os.path.isfile(filepath)
            assert writer.is_file_registered(filepath)

    for writer in correct_files_dict:
        assert writer.output_filepaths == correct_files_dict[writer]

    owa_md_filepath = os.path.join(output_dir, "class-A_file-2.md")
    output_a.delete_output_files(only_extension="md")
    assert not output_a.is_file_registered(owa_md_filepath)
    assert not os.path.isfile(owa_md_filepath)
    assert output_a.output_filepaths == [
        os.path.join(output_dir, "class-A_file-0.txt"),
        os.path.join(output_dir, "class-A_file-1.txt"),
    ]

    for writer in correct_files_dict:
        writer.delete_output_files()
        assert writer.output_filepaths == []
