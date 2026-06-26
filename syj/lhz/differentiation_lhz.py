# -*- coding: utf-8 -*-
"""兼容旧脚本名称的入口文件。

原来的长脚本已经拆分到main.py、env.py、trainer.py等模块中；保留本文件是为了
让旧命令python3 syj/lhz/differentiation_lhz.py仍然可以启动同一套训练流程。
"""

from main import main


if __name__ == "__main__":
    main()
