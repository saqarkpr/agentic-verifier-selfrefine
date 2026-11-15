"""Make `src/` importable as a flat package, the way the scripts import it.

Every script in src/ does `from verifier import verify`, because each is run
from inside src/ (`cd src && python train_lora_sft.py`). The tests keep that
import style rather than rewriting the scripts, so what the tests exercise is
exactly what runs.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
