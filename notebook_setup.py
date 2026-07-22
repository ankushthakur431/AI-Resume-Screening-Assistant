import os
import sys
from pathlib import Path

WORKSPACE = Path(r"d:/DATA_SCIENCE/FINAL_WORKBOOK/ML/DEPLOYMENT/GenAI&AgenticAI")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

os.chdir(WORKSPACE)
