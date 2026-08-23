# -*- coding: utf-8 -*-
"""一键依次运行监督损失和 Replay 两阶段长尾消融。"""

import sys
from pathlib import Path

ABLATIONS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABLATIONS_DIR))

from click_runner import run_tail_all


if __name__ == '__main__':
    run_tail_all(Path(__file__).resolve().parent)
