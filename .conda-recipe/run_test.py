#! /bin/bash
import sys
import pytest
import terrainbento

result = pytest.test()

if result == 0:
    sys.exit(0)
else:
    sys.exit(1)
