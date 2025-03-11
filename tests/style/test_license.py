import os
from pathlib import Path

import pytest

LICENSE_HEADER = """\
# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


def file_starts_with_license(file_path: str) -> bool:
    """Checks if the given file starts exactly with the expected license header."""
    with open(file_path, "r", encoding="utf-8") as f:
        file_start = f.read(len(LICENSE_HEADER))
    return file_start == LICENSE_HEADER


def find_python_files(directory: Path) -> list[Path]:
    """Recursively finds all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)
    return python_files


@pytest.mark.style
@pytest.mark.parametrize("file_path", find_python_files(Path(os.path.dirname(__file__)).parent.parent / "src"))
def test_python_file_has_license(file_path: str) -> None:
    """Test if the given Python file starts with the correct license header."""
    assert file_starts_with_license(file_path)
