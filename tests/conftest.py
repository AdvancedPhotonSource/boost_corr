"""Pytest configuration and fixtures."""
import sys
from unittest.mock import MagicMock

# Mock the timepix_dataset module before any test imports it
sys.modules['timepix_dataset'] = MagicMock()
sys.modules['timepix_dataset.dataset'] = MagicMock()
