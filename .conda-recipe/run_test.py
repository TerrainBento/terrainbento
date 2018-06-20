#! /bin/bash
import sys
import terrainbento

result = terrainbento.test()

if result.wasSuccessful():
    sys.exit(0)
else:
    sys.exit(1)
