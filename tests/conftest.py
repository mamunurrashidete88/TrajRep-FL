# tests/conftest.py
import sys
from pathlib import Path

# Root is on the path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))
