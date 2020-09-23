import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3

from util import get_args


def main(args):
    sis_df = pd.read_csv('./data/sis.csv')
    ling_fts = pickle.load(open('./ling_fts_new_df.p', 'rb'))

    bert_fail_idx = []
    ft_fail_idx = []
    hybrid_fail_idx = []

    for index, row in sis_df.iterrows():
        human_score = row['sentence average score']
        bert_score = row['bert prediction']
        ft_score = row['ft prediction']
        hybrid_score = row['hybrid prediction']
        if abs(human_score - bert_score) >= args.threshold:
            bert_fail_idx.append(index)
        if abs(human_score - ft_score) >= args.threshold:
            ft_fail_idx.append(index)
        if abs(human_score - hybrid_score) >= args.threshold:
            hybrid_fail_idx.append(index)

    set_bert = set(bert_fail_idx)
    set_ft = set(ft_fail_idx)
    set_hybrid = set(hybrid_fail_idx)
    set_100 = set_bert.difference(set_ft).difference(set_hybrid)
    set_010 = set_ft.difference(set_bert).difference(set_hybrid)
    set_001 = set_hybrid.difference(set_bert).difference(set_ft)
    set_110 = set_bert.intersection(set_ft).difference(set_hybrid)
    set_101 = set_bert.intersection(set_hybrid).difference(set_ft)
    set_011 = set_hybrid.intersection(set_ft).difference(set_bert)
    set_111 = set_bert.intersection(set_ft).intersection(set_hybrid)

    fig = venn3(
        subsets=(len(set_100), len(set_010), len(set_110), len(set_001), len(set_101), len(set_011), len(set_111)),
        set_labels=('BERT', 'FT', 'Hybrid'))
    venn_name = './figures/error_analysis_venn.pdf'
    plt.savefig(fig, venn_name)

    bert_fail_cases = sis_df.iloc[bert_fail_idx]
    ft_fail_cases = sis_df.iloc[ft_fail_idx]
    hybrid_fail_cases = sis_df.iloc[hybrid_fail_idx]

    bert_fail_cases.to_csv('./data/error_cases/bert_error_cases.csv', index=False)
    ft_fail_cases.to_csv('./data/error_cases/ft_error_cases.csv', index=False)
    hybrid_fail_cases.to_csv('./data/error_cases/hybrid_error_cases.csv', index=False)


if __name__ == '__main__':
    args = get_args()
    main(args)
