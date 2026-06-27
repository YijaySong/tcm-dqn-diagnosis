# -*- coding: utf-8 -*-
"""One-vs-Rest Random Forest 对比模型。"""

from argparse import ArgumentParser

from common import add_common_arguments, model_output_dir, probability_scores_from_estimator, run_score_model, skip_result

MODEL_SPEC = {
    'name': 'random_forest',
    'label': 'Random Forest OvR',
    'description': 'One-vs-Rest RandomForestClassifier非线性集成基线。',
    'category': 'ensemble',
}


def build_parser():
    parser = add_common_arguments(ArgumentParser(description='LHZ Random Forest OvR contrast model'))
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=-1)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = model_output_dir(MODEL_SPEC['name'], run_id=args.run_id, output_dir=args.output_dir)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multiclass import OneVsRestClassifier
    except ImportError:
        skip_result(output_dir, MODEL_SPEC, '当前环境未安装scikit-learn，跳过RandomForest。', config=vars(args))
        return

    def build_scores(x_train, y_train, x_test, dataset, logger):
        clf = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                class_weight='balanced_subsample',
            )
        )
        clf.fit(x_train, y_train)
        return probability_scores_from_estimator(clf, x_test)

    run_score_model(MODEL_SPEC, args, build_scores)


if __name__ == '__main__':
    main()
