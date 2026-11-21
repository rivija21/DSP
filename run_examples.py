"""Run every example script and regenerate examples/figures/*.png."""
import glob
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "examples"))
os.chdir(os.path.join(here, "examples"))
for path in sorted(glob.glob("ex*.py")):
    print(f"\n=== {path} ===")
    runpy.run_path(path, run_name="__main__")
