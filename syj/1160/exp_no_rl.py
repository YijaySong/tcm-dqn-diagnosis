# -*- coding: utf-8 -*-
"""无RL消融：运行纯监督基线的阈值模式。"""

import os
import runpy
import sys

if __name__ == '__main__':
    script_path = os.path.join(os.path.dirname(__file__), 'exp_supervised_baseline.py')
    sys.argv = [script_path, '--mode', 'threshold']
    runpy.run_path(script_path, run_name='__main__')
