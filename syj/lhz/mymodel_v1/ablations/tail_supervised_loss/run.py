# -*- coding: utf-8 -*-
"""打开本文件并点击 VS Code 右上角运行，即可完成本组正式消融。"""

import sys
from pathlib import Path

ABLATIONS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABLATIONS_DIR))

from click_runner import run_clickable


if __name__ == '__main__':
    run_clickable('tail_supervised_loss', Path(__file__).resolve().parent)
