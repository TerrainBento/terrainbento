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
    "requirements": ROOT / "requirements-extra",
    "root": ROOT,
}
PYTHON_VERSION = "3.12"


@nox.session(python=PYTHON_VERSION)
def test(session: nox.Session) -> None:
    """Run the tests."""
    session.install(".[testing]")

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


@nox.session(name="test-notebooks", python=PYTHON_VERSION)
def test_notebooks(session: nox.Session) -> None:
    """Test the notebooks."""
    session.install(".[testing,notebooks]")
    session.install("nbmake")

    args = [
        "notebooks",
        "--nbmake",
        "--nbmake-kernel=python3",
        "--nbmake-timeout=3000",
        "-vvv",
    ] + session.posargs

    session.run("pytest", *args)


@nox.session(name="build-docs")
def build_docs(session: nox.Session) -> None:
    """Build the docs."""
    session.install(".[docs]")

    PATH["build"].mkdir(exist_ok=True)
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "-W",
        "--keep-going",
        PATH["docs"],
        PATH["build"] / "html",
    )
    session.log(f"Generated docs at {PATH['build'] / 'html'!s}")


@nox.session
def lint(session: nox.Session) -> None:
    """Look for lint."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


@nox.session(name="sync-requirements", python=PYTHON_VERSION)
def sync_requirements(session: nox.Session) -> None:
    """Sync requirements files with pyproject.toml."""
    import tomllib

    PATH["requirements"].mkdir(exist_ok=True)

    with open(ROOT / "pyproject.toml", "rb") as fp:
        pyproject = tomllib.load(fp)

    _write_dependencies("requirements.txt", pyproject["project"]["dependencies"])
    for group, packages in pyproject["project"]["optional-dependencies"].items():
        _write_dependencies(
            PATH["requirements"] / f"requirements-{group}.txt", packages
        )


def _write_dependencies(filename, dependencies):
    with open(filename, "w") as fp:
        fp.write(f"{os.linesep.join(sorted(dependencies))}\n")


@nox.session
def build(session: nox.Session) -> None:
    """Build source and binary distributions."""
    session.install(".[build]")
    session.run("python", "-m", "build", "--outdir", PATH["dist"])
    session.run("twine", "check", PATH["dist"] / "*")


@nox.session
def release(session):
    """Tag, build and publish a new release to PyPI."""
    session.install(".[build]")
    session.run("fullrelease")


@nox.session(name="testpypi")
def publish_testpypi(session):
    """Upload package to TestPyPI."""
    build(session)
    session.run(
        "twine",
        "upload",
        "--skip-existing",
        "--repository-url",
        "https://test.pypi.org/legacy/",
        PATH["dist"] / "*",
    )


@nox.session(name="pypi")
def publish_pypi(session):
    """Upload package to PyPI."""
    build(session)
    session.run(
        "twine",
        "upload",
        "--skip-existing",
        "--repository-url",
        "https://upload.pypi.org/legacy/",
        PATH["dist"] / "*",
    )


@nox.session(python=False)
def clean(session):
    """Remove virtual environments, build files, and caches."""
    shutil.rmtree(PATH["build"], ignore_errors=True)
    shutil.rmtree(PATH["dist"], ignore_errors=True)
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
    shutil.rmtree(PATH["nox"], ignore_errors=True)
