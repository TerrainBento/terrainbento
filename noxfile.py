import os
import pathlib
import shutil
from itertools import chain

import nox

PROJECT = "terrainbento"
HERE = pathlib.Path(__file__)
ROOT = HERE.parent
PATH = {
    "build": ROOT / "build",
    "dist": ROOT / "dist",
    "docs": ROOT / "docs",
    "notebooks": ROOT / "notebooks",
    "nox": pathlib.Path(".nox"),
    "root": ROOT,
}
PYTHON_VERSION = "3.12"


@nox.session(python=PYTHON_VERSION)
def test(session: nox.Session) -> None:
    """Run the tests."""
    session.install(".[testing,notebooks]")

    args = [
        "--cov",
        PROJECT,
        "-vvv",
    ] + session.posargs

    if "CI" in os.environ:
        args.append(f"--cov-report=xml:{ROOT.absolute()!s}/coverage.xml")
    session.run("pytest", *args)

    if "CI" not in os.environ:
        session.run("coverage", "report", "--ignore-errors", "--show-missing")


@nox.session(python=False)
def clean(session):
    """Remove virtual environments, build files, and caches."""
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("docs/build", ignore_errors=True)
    shutil.rmtree(f"{PROJECT}.egg-info", ignore_errors=True)
    shutil.rmtree(".pytest_cache", ignore_errors=True)
    shutil.rmtree(".venv", ignore_errors=True)
    if os.path.exists(".coverage"):
        os.remove(".coverage")
    for p in chain(ROOT.rglob("*.py[co]"), ROOT.rglob("__pycache__")):
        if p.is_dir():
            p.rmdir()
        else:
            p.unlink()


@nox.session(python=False)
def nuke(session):
    """Clean and also remove the .nox/ directory."""
    clean(session)
    shutil.rmtree(".nox", ignore_errors=True)
