# -*- coding: utf-8 -*-
"""V1 推理共享核心；直接复用训练端 model/env/state，避免双实现漂移。"""

import re
from pathlib import Path

import torch

from constants import device
from env import Environment
from model import build_q_network
from state import build_state_vector, predict_actions_from_state


RUNS_DIR = Path(__file__).resolve().parent / 'runs'


def normalize_symptoms(text):
    return [item.strip() for item in re.split(r'[,，\s]+', text) if item.strip()]


def find_latest_checkpoint(runs_dir=RUNS_DIR):
    """查找最近生成的 best.pt，支持普通 run 和多 seed 子目录。"""
    candidates = [path for path in Path(runs_dir).rglob('best.pt') if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f'{runs_dir} 下没有可用的 best.pt；请先完成训练')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_recommender(model_path=None):
    path = Path(model_path) if model_path else find_latest_checkpoint()
    if not path.exists():
        raise FileNotFoundError(f'未找到 V1 checkpoint: {path}')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('checkpoint_version') != 3:
        raise ValueError('仅支持 V1 checkpoint_version=3，不支持无词表的旧 state_dict')
    env = Environment(checkpoint['symptoms'], checkpoint['Se'])
    config = checkpoint['config']
    env.configure_reward(config['reward_beta'], config['step_cost'], config['stop_utility_scale'], config['gamma'])
    model = build_q_network(checkpoint['model_type'], len(env.state_space), len(env.action_space),
                            checkpoint['nn_units'], checkpoint['nn_units2'], checkpoint['dropout']).to(device)
    model.load_state_dict(checkpoint['policy_state_dict'])
    model.eval()
    return model, env, {'model_path': str(path), 'config': config, 'validation_metrics': checkpoint.get('validation_metrics', {})}


def predict_symptoms(model, env, symptoms, min_actions=1, stop_margin_threshold=None):
    state = build_state_vector(env, symptoms)
    actions = predict_actions_from_state(env, model, device, state, min_actions=min_actions,
                                         stop_margin_threshold=stop_margin_threshold)
    return [env.action_space[action] for action in actions]


def predict_symptoms_trace(model, env, symptoms, candidate_topk=3, min_actions=1,
                           stop_margin_threshold=None):
    state = build_state_vector(env, symptoms)
    actions, trace = predict_actions_from_state(
        env, model, device, state, min_actions=min_actions,
        stop_margin_threshold=stop_margin_threshold,
        return_trace=True, candidate_topk=candidate_topk,
    )
    return {'recommendations': [env.action_space[action] for action in actions], 'trace': trace, 'candidate_topk': candidate_topk}


def print_trace_prediction(result, symptoms):
    selected = []
    print(f'  输入刻下症: {", ".join(symptoms)}')
    print('\n  === DQN辨证过程 ===')
    for step, item in enumerate(result['trace'], start=1):
        print(f'  第{step}步:')
        print(f'    当前已选证候要素: {", ".join(selected) if selected else "无"}')
        print(f'    模型选择: {item["action_name"]}')
        candidates = item.get('candidates', [])
        if candidates:
            ranking = ', '.join(
                f'{candidate["action_name"]}({candidate["q_value"]:.4f})'
                for candidate in candidates
            )
            print(f'    候选排名: {ranking}')
        if not item.get('is_stop', False):
            selected.append(item['action_name'])
        print()
    print(f'  最终推荐证候要素: {", ".join(result["recommendations"]) if result["recommendations"] else "无"}')


def main():
    """直接点击运行共享模块时，使用最新模型执行内置示例。"""
    symptoms = normalize_symptoms('胸闷,胸痛,畏寒')
    model, env, metadata = load_recommender()
    known = [symptom for symptom in symptoms if symptom in env.symptom_to_index]
    config = metadata['config']
    print(f'自动加载最新模型: {metadata["model_path"]}')
    print(f'输入症状: {", ".join(known)}')
    recommendations = predict_symptoms(
        model, env, known,
        min_actions=config.get('min_actions_before_stop', 1),
        stop_margin_threshold=config.get('stop_margin_threshold'),
    )
    print(f'推荐证候要素: {", ".join(recommendations)}')


if __name__ == '__main__':
    main()
