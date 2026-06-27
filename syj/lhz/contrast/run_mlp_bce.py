# -*- coding: utf-8 -*-
"""MLP-BCE监督多标签对比模型。"""

from argparse import ArgumentParser

import numpy as np

from common import add_common_arguments, run_score_model

MODEL_SPEC = {
    'name': 'mlp_bce',
    'label': 'MLP-BCE',
    'description': '使用症状multi-hot输入和BCEWithLogitsLoss训练的监督多标签神经网络基线。',
    'category': 'neural_baseline',
}


def build_parser():
    parser = add_common_arguments(ArgumentParser(description='LHZ MLP-BCE supervised contrast model'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--hidden-dim2', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    return parser


def main():
    args = build_parser().parse_args()

    def build_scores(x_train, y_train, x_test, dataset, logger):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )
        logger.info(f'可用设备:{device}')

        class MLPClassifier(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, args.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.hidden_dim, args.hidden_dim2),
                    nn.ReLU(),
                    nn.Linear(args.hidden_dim2, output_dim),
                )

            def forward(self, x):
                return self.net(x)

        model = MLPClassifier(x_train.shape[1], y_train.shape[1]).to(device)
        x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

        pos_count = y_train_t.sum(dim=0)
        neg_count = y_train_t.shape[0] - pos_count
        pos_weight = (neg_count / torch.clamp(pos_count, min=1.0)).clamp(min=1.0, max=10.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
        batch_size = min(args.batch_size, len(x_train_t))

        model.train()
        for epoch in range(args.epochs):
            permutation = torch.randperm(len(x_train_t), device=device)
            total_loss = 0.0
            for start in range(0, len(x_train_t), batch_size):
                idx = permutation[start:start + batch_size]
                logits = model(x_train_t[idx])
                loss = criterion(logits, y_train_t[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(idx)
            logger.info(f'MLP-BCE epoch {epoch + 1}/{args.epochs}, loss={total_loss / len(x_train_t):.6f}')

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32, device=device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.asarray(probs, dtype=float)

    run_score_model(MODEL_SPEC, args, build_scores)


if __name__ == '__main__':
    main()
