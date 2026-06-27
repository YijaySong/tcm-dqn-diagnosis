# -*- coding: utf-8 -*-
"""One-vs-Rest Bernoulli Naive Bayes 对比模型。"""

from argparse import ArgumentParser

from common import add_common_arguments, model_output_dir, probability_scores_from_estimator, run_score_model, skip_result

MODEL_SPEC = {
    'name': 'bernoulli_nb',
    'label': 'Bernoulli NB',
    'description': '适合二值症状特征的Bernoulli Naive Bayes多标签基线。',
    'category': 'classical',
}


def build_parser():
    parser = add_common_arguments(ArgumentParser(description='LHZ BernoulliNB OvR contrast model'))
    parser.add_argument('--alpha', type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = model_output_dir(MODEL_SPEC['name'], run_id=args.run_id, output_dir=args.output_dir)
    try:
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.multiclass import OneVsRestClassifier
    except ImportError:
        skip_result(output_dir, MODEL_SPEC, '当前环境未安装scikit-learn，跳过BernoulliNB。', config=vars(args))
        return

    def build_scores(x_train, y_train, x_test, dataset, logger):
        clf = OneVsRestClassifier(BernoulliNB(alpha=args.alpha))
        clf.fit(x_train, y_train)
        return probability_scores_from_estimator(clf, x_test)

    run_score_model(MODEL_SPEC, args, build_scores)


if __name__ == '__main__':
    main()
