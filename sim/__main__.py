"""命令行环境自检入口：python -m sim。"""

from __future__ import annotations

import sys

from sim.selfcheck import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
