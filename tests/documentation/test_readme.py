import os
import re
from pathlib import Path

import pytest

from ..common import run_script_successfully

README_PATH = Path(os.path.dirname(__file__)).parent.parent / "README.md"
SCRIPT_PATH = "test_readme_snippets.py"


def extract_python_code_blocks(readme_path: Path) -> str:
    """Extracts all Python code blocks from a Markdown README file."""
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match fenced code blocks marked as python (```python ... ```)
    code_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    return "\n\n".join(code_blocks)  # Concatenate all code snippets


@pytest.mark.cuda
def test_readme_code_blocks() -> None:
    """Writes all extracted Python code blocks to a single file, lints it with flake8, and executes it."""
    code = extract_python_code_blocks(README_PATH)

    with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(code)

    # Run flake8 linting
    run_script_successfully(SCRIPT_PATH)
