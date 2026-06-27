# -*- coding: utf-8 -*-
"""One-vs-Rest HistGradientBoosting 对比模型。"""

from argparse import ArgumentParser

from common import add_common_arguments, model_output_dir, probability_scores_from_estimator, run_score_model, skip_result

MODEL_SPEC = {
    'name': 'hist_gradient_boosting',
    'label': 'HistGradientBoosting OvR',
    'description': 'One-vs-Rest HistGradientBoostingClassifier梯度提升对比模型。',
    'category': 'modern_ml',
}


def build_parser():
    parser = add_common_arguments(ArgumentParser(description='LHZ HistGradientBoosting OvR contrast model'))
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--max-leaf-nodes', type=int, default=31)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = model_output_dir(MODEL_SPEC['name'], run_id=args.run_id, output_dir=args.output_dir)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.multiclass import OneVsRestClassifier
    except ImportError:
        skip_result(output_dir, MODEL_SPEC, '当前环境未安装scikit-learn或版本不支持HistGradientBoostingClassifier，跳过该模型。', config=vars(args))
        return

    def build_scores(x_train, y_train, x_test, dataset, logger):
        clf = OneVsRestClassifier(
            HistGradientBoostingClassifier(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                max_leaf_nodes=args.max_leaf_nodes,
                random_state=args.seed,
            )
        )
        clf.fit(x_train, y_train)
        return probability_scores_from_estimator(clf, x_test)

    run_score_model(MODEL_SPEC, args, build_scores)


if __name__ == '__main__':
    main()
