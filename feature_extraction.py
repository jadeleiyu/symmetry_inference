import pickle

import pandas as pd

from encoders import FeatureBasedEncoder
from util import get_args

feature_names = ["trans", "trans_mod", "v_tense", "v_act", "modal", "neg",
                 "is_root",
                 "direction", "sing_sub", "sing_obj",
                 "conj_sub", "conj_obj", "rcp_phrase", "ani_sub", "ani_match",
                 "sub_more_freq", "num_np", "num_clauses"]


def main(args):
    encoder = FeatureBasedEncoder()
    sis_file = args.sis_file
    sis_feature_file = args.sis_feature_file
    sis_df = pd.read_csv(sis_file, encoding='latin1')
    sentences = list(sis_df['sentence'])
    verbs = list(sis_df['predicate'])
    features = encoder.encode(sentences, verbs)
    num_sents = features.shape[0]
    assert len(feature_names) == features.shape[-1]
    features_df = {}
    for k in range(len(feature_names)):
        ft_name = feature_names[k]
        features_df[ft_name] = [features[i][k] for i in range(num_sents)]
    features_df = pd.DataFrame(features_df)
    features_df['sentence'] = sis_df['sentence']
    features_df['predicate'] = sis_df['predicate']
    features_df['pred_id'] = sis_df['pred_id']
    features_df['raw_id'] = sis_df['raw_id']
    features_df['rating_type'] = sis_df['rating_type']
    features_df['verb symmetry score'] = sis_df['rating']
    features_df.to_csv(sis_feature_file, index=False)
    pickle.dump(features, open('./data/sis_ling_fts.p', 'wb'))


if __name__ == '__main__':
    args = get_args()
    main(args)
