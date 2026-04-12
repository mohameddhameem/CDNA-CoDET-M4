"""Project root path helpers.

Notes
-----
Defines the resolved repository root directory.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()