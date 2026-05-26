#!/usr/bin/env python
from pathlib import Path
import sys


path = Path(sys.argv[1])
lines = path.read_text(errors="replace").splitlines()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
for i, line in enumerate(lines[:n], 1):
    print(i, repr(line))
