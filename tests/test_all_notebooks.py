import os
import subprocess
import tempfile

import nbformat

_TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "notebooks")
)
_EXCLUDE = []


def all_notebooks(path="."):
    notebooks = []
    for root, _, files in os.walk(path):
        if ".ipynb_checkpoints" not in root:
            for file in files:
                if file.endswith(".ipynb") and (file not in _EXCLUDE):
                    notebooks.append(os.path.join(root, file))
    return notebooks


def pytest_generate_tests(metafunc):
    if "notebook" in metafunc.fixturenames:
        metafunc.parametrize("notebook", all_notebooks(_TEST_DIR))


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
    :returns (parsed nb object, execution errors)
    """
    with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as fp:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.kernel_name=python",
            "--ExecutePreprocessor.timeout=None",
            "--output",
            fp.name,
            "--output-dir=.",
            path,
        ]
        subprocess.check_call(args)

        nb = nbformat.read(
            fp.name,
            nbformat.current_nbformat,
            encoding="UTF-8"  # See below
            # If you are getting a TypeError because __init__ got an unexpected
            # keyword argument "encoding", it's probably because you are using
            # a version of python that is too advanced. This function call
            # works in 3.7, but not 3.9. HOWEVER, looking at the nbformat and
            # json source code, the "encoding" kwarg is never actually used
            # along the way to the failure point and nbformat.read hardcodes
            # "utf-8". Could probably delete?
            #
            # Use the following if it breaks:
            # fp.name, nbformat.current_nbformat
        )

    errors = [
        output
        for cell in nb.cells
        if "outputs" in cell
        for output in cell["outputs"]
        if output.output_type == "error"
    ]

    return nb, errors


def test_notebook(tmpdir, notebook):
    with tmpdir.as_cwd():
        _, errors = _notebook_run(notebook)
        assert not errors
