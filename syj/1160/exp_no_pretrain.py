# -*- coding: utf-8 -*-
"""无预训练消融：关闭监督预训练后运行DQN。"""

import os
import runpy
import sys

if __name__ == '__main__':
    script_path = os.path.join(os.path.dirname(__file__), 'differentiation.py')
    sys.argv = [
        script_path,
        '-use_pretrain', '0',
        '-replay_warmup', '1',
    ]
    runpy.run_path(script_path, run_name='__main__')
