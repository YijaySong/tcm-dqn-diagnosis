# -*- coding: utf-8 -*-
"""运行 V1 的一个消融实验族。"""

import argparse

from common import FAMILIES, run_family


def main():
    parser = argparse.ArgumentParser(description='运行 mymodel_v1 消融实验族')
    parser.add_argument('family', choices=sorted(FAMILIES))
    parser.add_argument('--seeds', default='9,17,29')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--dry-run', action='store_true')
    # ``parse_known_args`` 允许运行器自己的选项自然地放在 family 后面，未知参数
    # 原样转交 main.py；此前 ``REMAINDER`` 会把 --dry-run 误转交给 main.py。
    args, main_args = parser.parse_known_args()
    extras = main_args[1:] if main_args[:1] == ['--'] else main_args
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    directory, records = run_family(args.family, seeds, extras, args.run_id, args.dry_run)
    print(f'结果目录: {directory}')
    for item in records:
        print(item['experiment'], item['seed'], item['status'])


if __name__ == '__main__':
    main()
