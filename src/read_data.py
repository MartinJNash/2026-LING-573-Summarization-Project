# Compatibility shim — re-exports from top-level read_data.py.
# Martin's code uses `from src.read_data import ...`; this keeps that working.
from read_data import (
    read_gs_training_data,
    read_large_scale_training_data,
    read_test_training_data,
    read_folder,
)
