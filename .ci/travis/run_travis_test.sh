#! /bin/bash

PYTHON=${PYTHON:-python}

run_test()
{
  mkdir -p _test
  cd _test

  INSTALLDIR=$($PYTHON -c "import os; import terrainbento; print(os.path.dirname(terrainbento.__file__))")

  coverage run --source=$INSTALLDIR ../scripts/test-installed-terrainbento.py && (coverage report && cp .coverage ..)
}

run_test
