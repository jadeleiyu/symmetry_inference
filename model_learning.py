import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from tqdm import tqdm

from encoders import *
from util import get_args


def main(args):

    # initialize the encoder
    if args.model_type == 'static':
        if args.embedding == 'word2vec':
            encoder = Word2vecEncoder
        if args.embedding == 'glove':
            encoder = GloveEncoder
    if args.model_type == 'ft':
        encoder = FeatureBasedEncoder()
    if args.model_type == 'bert' and not args.fine_tune:
        encoder = BertEncoder()
    else:
        raise NotImplementedError

    sis_df = pd.read_csv('./data/sis.csv', encoding='latin1')
    sentences = list(sis_df['sentence'])
    pred_ids = list(set(list(sis_df['pred_id'])))
    predicates = list(sis_df['predicate'])
    correlations = []
    X = encoder.encode(sentences, verbs=predicates)
    y_evals_true = []
    y_evals_pred = []
    for pred_id in tqdm(pred_ids):
        train_sents_idx = sis_df.index[sis_df['pred_id'] != pred_id].tolist()
        eval_sents_idx = sis_df.index[sis_df['pred_id'] == pred_id].tolist()

        X_train = X[train_sents_idx]
        X_eval = X[eval_sents_idx]
        y_train = list(sis_df.loc[sis_df['pred_id'] != pred_id]['sentence average score'])
        y_eval = list(sis_df.loc[sis_df['pred_id'] == pred_id]['sentence average score'])

        mlr_model = lr_model_training(X_train, y_train, alpha=args.ridge_alpha)
        y_pred = mlr_model.predict(X_eval)
        correlation, p_val = pearsonr(y_pred, y_eval)
        correlations.append(correlation)
        y_evals_true += y_eval
        y_evals_pred += list(y_pred)

    mse = np.square(np.array(y_evals_true) - np.array(y_evals_pred)).mean()
    correlation_all = pearsonr(np.array(y_evals_true), np.array(y_evals_pred))
    print("Training {} model is complete.".format(args.model_type))
    print("Mean square error between model prediction and human judgement: {}".format(mse))
    print("Correlation between model prediction and human judgement: {}".format(correlation_all))
    # save predictions to sis dataframe
    prediction_score_name = args.model_type + ' prediction'
    sis_df[prediction_score_name] = pd.Series(y_evals_pred)
    return correlations, correlation_all, y_evals_pred


def lr_model_training(X_cont, y_cont, alpha=1.0):
    lr_model = Ridge(alpha=alpha)
    lr_model.fit(X_cont, y_cont)
    return lr_model


if __name__ == '__main__':
    args = get_args()
    correlations, correlation_all, y_evals_pred = main(args)
