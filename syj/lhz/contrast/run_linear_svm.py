# -*- coding: utf-8 -*-
"""One-vs-Rest Linear SVM 对比模型。"""

from argparse import ArgumentParser

from common import add_common_arguments, model_output_dir, probability_scores_from_estimator, run_score_model, skip_result

MODEL_SPEC = {
    'name': 'linear_svm',
    'label': 'Linear SVM OvR',
    'description': 'One-vs-Rest LinearSVC多标签最大间隔基线。',
    'category': 'classical',
}


def build_parser():
    parser = add_common_arguments(ArgumentParser(description='LHZ Linear SVM OvR contrast model'))
    parser.set_defaults(threshold=0.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--max-iter', type=int, default=5000)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = model_output_dir(MODEL_SPEC['name'], run_id=args.run_id, output_dir=args.output_dir)
    try:
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC
    except ImportError:
        skip_result(output_dir, MODEL_SPEC, '当前环境未安装scikit-learn，跳过LinearSVC。', config=vars(args))
        return

    def build_scores(x_train, y_train, x_test, dataset, logger):
        clf = OneVsRestClassifier(LinearSVC(C=args.c, max_iter=args.max_iter, random_state=args.seed))
        clf.fit(x_train, y_train)
        return probability_scores_from_estimator(clf, x_test)

    run_score_model(MODEL_SPEC, args, build_scores, threshold_is_margin=True)


if __name__ == '__main__':
    main()
