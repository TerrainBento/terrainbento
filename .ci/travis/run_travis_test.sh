#! /bin/bash

PYTHON=${PYTHON:-python}

run_test()
{
  mkdir -p _test
  cd _test

  INSTALLDIR=$($PYTHON -c "import os; import terrainbento; print(os.path.dirname(terrainbento.__file__))")

  python ../scripts/test-installed-terrainbento.py && (cp .coverage ..)
}

run_test
