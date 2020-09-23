import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_tuning', action="store_true",
                        help="whether to fine tune BERT parameters during training")
    parser.add_argument('--analysis', default='verb_level', type=str,
                        help="whether to perform error analysis at verb level or sentence level")
    parser.add_argument('--threshold', default=1.0, type=float,
                        help="threshold difference between predicted score and true score")
    parser.add_argument('--model_type', default='ft', type=str)
    parser.add_argument('--embedding', default='word2vec', type=str)
    parser.add_argument('--sis_file', default='./data/sis.csv', type=str)
    parser.add_argument('--sis_feature_file', default='./data/sis_with_features.csv', type=str)
    parser.add_argument('--cuda', action="store_true",
                        help='whether to use gpu accelerating')
    parser.add_argument('--ridge_alpha', default=1.0, type=float)
    return parser.parse_args()
