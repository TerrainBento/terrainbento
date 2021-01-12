# coding: utf8
# !/usr/env/python

import pytest
from .conftest import clock_08

from terrainbento.output_writers import GenericOutputWriter

class EmptyModel:
    def __init__(self):
        self.clock = clock_08

@pytest.mark.parametrize("in_name, add_id, out_name", [
    (None, True, "output-writer"),
    (None, False, "output-writer"),
    ('given_nameT', True, "given_nameT"),
    ('given_nameF', False, "given_nameF"),
    ])
def test_names(in_name, add_id, out_name):
    # Make a few writer to check if the id value is handled correctly
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
