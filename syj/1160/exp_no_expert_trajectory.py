# -*- coding: utf-8 -*-
"""无专家轨迹消融：关闭ReplayMemory专家预填充后运行DQN。"""

import os
import runpy
import sys

if __name__ == '__main__':
    script_path = os.path.join(os.path.dirname(__file__), 'differentiation.py')
    sys.argv = [
        script_path,
        '-use_pretrain', '1',
        '-replay_warmup', '0',
    ]
    runpy.run_path(script_path, run_name='__main__')
