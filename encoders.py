import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from transformers import *
import clausiepy
import getngrams
from nltk.tag.stanford import StanfordNERTagger
import zipfile
import embed_helper
import gensim
import gensim.downloader as api
import gensim.models.keyedvectors as word2vec


class Encoder:
    def __init__(self):
        pass

    def encode(self, sentences, verbs):
        pass


class Word2vecEncoder(Encoder):
    def __init__(self, embedding_fn="./static_embeddings/GoogleNews-vectors-negative300.bin.gz"):
        super().__init__()

        print("Loading word2vec model...")
        self.model = word2vec.KeyedVectors.load_word2vec_format(embedding_fn,
                                                                binary=True)
        self.model.init_sims(replace=True)
        print("w2v model loaded.")

    def encode(self, sentences, verbs=None):
        embed_helper.tokenize_text(sentences)
        return embed_helper.word_avg_list(self.model, sentences)


class GloveEncoder(Encoder):
    def __init__(self, embedding_fn="./static_embeddings/glove.6B/glove.6B.300d.txt"):
        super().__init__()

        print("Loading Glove model...")
        self.model = embed_helper.load(embedding_fn)
        # glove_file = ("./glove.6B.300d")
        # self.model = embed_helper.load_glove_model(glove_file)
        print("Glove model loaded.")

    def encode(self, sentences, verbs=None):
        embed_helper.tokenize_text(sentences)
        return embed_helper.word_avg_list(self.model, sentences)


class SbertEncoder(Encoder):
    def __init__(self, pretrained_weights='bert-base-nli-mean-tokens'):
        super().__init__()

        self.model = SentenceTransformer(pretrained_weights)

    def encode(self, sentences, verbs=None):
        return np.array(self.model.encode(sentences))


class BertEncoder(Encoder):
    def __init__(self, pretrained_weights='bert-base-uncased'):
        super().__init__()
        self.model_class = BertModel
        self.tokenizer_class = BertTokenizer
        self.pretrained_weights = pretrained_weights
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.model = self.model_class.from_pretrained(pretrained_weights)

    def encode(self, sentences, verbs=None):
        X = []
        for sentence in sentences:
            input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)])
            with torch.no_grad():
                hidden_states = self.model(input_ids)
            all_hidden_states, all_attentions = hidden_states[-2:]
            # take last hidden states for each word, and take average
            last_states = all_hidden_states[0].squeeze()
            avg_last_states = torch.mean(last_states, 0)
            X.append(avg_last_states)
        return torch.stack(X).numpy()


class FeatureBasedEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

        jar = 'stanford-ner-4.0.0/stanford-ner.jar'
        model = 'stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz'

        self.st = StanfordNERTagger(model, jar)

        self.present_tense = ['VB', 'VBG', 'VBP', 'VBZ']
        self.noun_types = ['NOUN', 'PROPN', 'PRON']
        self.preps = ['with', 'up', 'off', 'from']
        self.modals = ['might', 'can', 'could', 'may']
        self.dt_list = ['mix', 'compare', 'combine']
        self.personal_sg = ["i", "he", "she", "him", "her", "hers", "herself", "himself", "hisself",
                            "it", "itself", "me", "myself", "one", "oneself", "ownself"]
        self.personal_plu = ["we", "us", "they", "them", "our", "ours", "theirs", "ourselves", "themselves"]

    def number(self, word, roles):
        """ Determines whether noun is singular or plural (conjoined counts as plural)
        """
        explain = spacy.explain(word.tag_)

        if word.pos_ == "NOUN":
            if 'singular' in explain:
                return 1
            elif 'plural' in explain:
                return 0
            else:
                return 2

        elif word.pos_ == "PROPN":
            if word.text in roles:
                if 'PERSON' in roles[word.text]:
                    return 1
                else:
                    return 0

        elif word.pos_ == "PRON":
            if word.text.lower() in self.personal_sg:
                return 1
            elif word.text.lower() in self.personal_plu:
                return 0
            else:
                return 2
        else:
            return 2

    def tense(self, verb):
        """ Determines whether verb is present tense or not
        """
        if str(verb.tag_) in self.present_tense or 'base form' in spacy.explain(verb.tag_):
            return 1
        else:
            return 0

    def get_ner_roles(self, sent):
        """ Returns entity types for each argument """
        import string
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        codes = self.st.tag(sent.split())

        def tup_to_dict(tup, di):
            for a, b in tup:
                di.setdefault(a, []).append(b)
            return di

        dictionary = {}
        ner_roles = tup_to_dict(codes, dictionary)

        return ner_roles

    def get_freq(self, r1, r2):
        """ Compare usage frequency between arguments """

        s1 = 0
        s2 = 0
        params = '-alldata,-startYear=1900,-endYear=2012,-caseInsensitive,-noprint,-nosave'

        d1 = getngrams.runQuery(r1 + ',' + params)
        i = 1
        if len(d1.columns) > 1:
            s1 = np.sum(d1[r1])

        d2 = getngrams.runQuery(r2 + ',' + params)
        if len(d2.columns) > 1:
            s2 = np.sum(d2[r2])

        if s1 > s2:
            return 1
        else:
            return 0

    def check_pron(self, word, verb, roles):
        """ If argument is a pronoun, switch to its subject form """
        w = ''
        v = ''
        if word.pos_ == "PRON":
            if word.text.lower() == "me":
                w = "I"
            elif word.text.lower() in ["him", "his", "himself"]:
                w = "he"
            elif word.text.lower() in ["her", "hers", "herself"]:
                w = "she"
            elif word.text.lower() == ["them", "theirs", "themselves"]:
                w = "they"
            elif word.text.lower() == ["us", "ours", "ourselves"]:
                w = "we"
            elif word.text.lower() == ["it"]:
                w = "it"
            else:
                w = word.text

        else:
            w = word.text

        v = self.sub_verb_agree(self.nlp(word.text), verb, roles)

        return w + " " + v

    def sub_verb_agree(self, word, verb, roles):
        """ Return consistent subject-verb form """
        v = str(verb.lemma_)

        irr_past = {"see": "saw", "break": "broke", "eat": "ate", "know": "knew"}

        if self.tense(verb):  # if present tense
            if self.number(word[0], roles) != 0 and word[0].text.lower not in ["I", "you"]:
                if v in ["mix", "match", "clash"]:
                    return v + 'es'
                elif v in ["copy", "hurry", "marry"]:
                    return v[:-1] + 'ies'
                else:
                    return v + 's'
            else:
                return v

        else:
            if v in ["mix", "match", "clash", "kill", "applaud", "differ", "drown"]:
                return v + 'ed'
            elif v in ["copy", "hurry", "marry"]:
                return v[:-1] + 'ied'
            elif v in irr_past:
                return irr_past[v]
            else:
                return v + 'd'

    def check_ani_match(self, arg1, arg2, roles):
        """ Check if entity types match """
        a1 = arg1.text
        a2 = arg2.text
        if a1 in roles and a2 in roles:
            if roles[a1][0] == roles[a2][0]:
                return 1
            elif arg1.pos_ == "PRON" and arg2.pos_ == "PRON":
                return 1
            elif (arg1.pos_ == "PRON" and arg2.pos_ == "PROPN") or (arg1.pos_ == "PROPN" and arg2.pos_ == "PRON"):
                return 1
            elif (arg1.pos_ == "PRON" and arg2.pos_ != "PROPN") or (arg1.pos_ == "PROPN" and arg2.pos_ != "PRON"):
                return 0
            else:
                return 0

    def check_ani(self, arg, roles):
        """ Check if entity types match """
        a = arg.text
        if a in roles:
            if arg.pos_ in ['PRON', 'PROPN'] or roles[a][0] in ['PERSON', 'ORGANIZATION']:
                return 1
            else:
                return 0
        else:
            return 0

    def check_activity(self, verb):
        """ Check for verb activity """
        list_active = ['marry',
                       'meet',
                       'combine',
                       'mix',
                       'argue',
                       'compare',
                       'separate',
                       'copy',
                       'love',
                       'hate',
                       'save',
                       'see',
                       'hit',
                       'kill',
                       'bounce',
                       'lecture',
                       'hurry',
                       'chase',
                       'hurt',
                       'push',
                       'applaud',
                       'follow',
                       'eat',
                       'drown',
                       'choke',
                       'chat',
                       'clash',
                       'converse',
                       'break',
                       'communicate',
                       'collaborate']

        list_stative = ['know',
                        'resemble',
                        'differ',
                        'match',
                        'tie',
                        'rhyme',
                        'agree',
                        'alternate',
                        'coexist']

        if verb.text in list_active:
            return 1
        else:
            return 0

    def get_clauses(self, sent):

        clauses = clausiepy.clausie(sent)

        num_np = 0
        for clause in clauses:
            if 'S' in clause:
                if len(clause['S']) > 0:
                    num_np = num_np + 1
                    for key in clause['S']:
                        for t in key.rights:
                            if t.pos_ in ["NOUN", "PROPN", "PRON"]:
                                num_np = num_np + 1

            if 'O' in clause:
                if len(clause['O']) > 0:
                    num_np = num_np + 1

                    for key in clause['O']:
                        for t in key.rights:
                            if t.pos_ in ["NOUN", "PROPN", "PRON"]:
                                num_np = num_np + 1

            if 'V' in clause:
                if len(clause['V']) > 0:

                    for key in clause['V']:
                        for t in key.rights:
                            if t.dep_ in ["NOUN", "PROPN", "PRON"]:
                                num_np = num_np + 1

            if 'A' in clause:
                if len(clause['A']) > 0:

                    for key in clause['A']:
                        for t in key.rights:
                            if t.dep_ == 'pobj':
                                num_np = num_np + 1

        return len(clauses), num_np

    def get_features(self, sent):
        clauses = clausiepy.clausie(sent)
        ner_roles = self.get_ner_roles(sent)

        num_clauses, num_np = self.get_clauses(sent)

        i = 0

        trans = 0
        trans_mod = 0
        v_tense = 0

        is_root = 0
        direction = 0
        sub_exists = 0
        plu_num_np = 0
        sing_sub = 0
        sing_obj = 0
        conj_sub = 0
        conj_obj = 0
        ani_sub = 0

        v_act = 0
        modal = 0
        neg = 0
        ani_match = 0
        sub_more_freq = 0

        r1 = ''
        r2 = ''
        r1a = ''
        r2a = ''

        v = ''

        for clause in clauses:

            for key in clause:

                if key[i] == "S" and len(clause[key]) > 0:

                    r1 = clause[key][i]
                    sub_exists = 1

                    ani_sub = self.check_ani(r1, ner_roles)

                    for t in clause[key][i].rights:
                        if t.pos_ == 'CCONJ':
                            conj_sub = 1
                            plu_num_np = 1
                            sing_sub = 0

                        elif t.pos_ in self.noun_types:
                            r2 = t

                    if conj_sub < 1:
                        sing_sub = self.number(clause[key][i], ner_roles)
                    else:
                        sing_sub = 0

                elif key[i] == "O" and len(clause[key]) > 0:

                    plu_num_np = 1
                    direction = 1

                    r2 = clause[key][i]
                    if 'CCONJ' in [t.pos_ for t in clause[key][i].rights]:
                        conj_obj = 1
                        plu_num_np = 1
                        sing_obj = 0
                    else:
                        sing_obj = self.number(clause[key][i], ner_roles)

                elif key[i] == "A":

                    if len(clause[key]) > 0:

                        for p in clause[key]:
                            if str(p) in self.preps and p.head in clause['V']:

                                for t in p.rights:
                                    if t.dep_ == 'pobj':
                                        r2 = t
                                        trans_mod = 1
                                        trans = 0
                                        direction = 1


                elif key[i] == "V" and len(clause[key]) > 0:

                    v = clause[key][i]
                    v_tense = self.tense(v)

                    for t in clause[key][i].lefts:
                        if "S" in clause:
                            if t in clause['S'] or 'ROOT' in t.dep_:
                                is_root = 1
                                v_act = self.check_activity(v)

                        if 'modal' in spacy.explain(t.tag_):
                            if t.text in self.modals:
                                modal = 1

                        elif 'neg' in t.dep_:
                            neg = 1

                    for t in clause[key][i].rights:
                        if t.dep_ == "dobj":
                            trans = 1

                            if str(v.lemma_) in self.dt_list:
                                r1 = t
                                ani_sub = self.check_ani(t, ner_roles)
                                for j in t.rights:
                                    if j.pos_ == "NOUN":
                                        r2 = j

            break

        if len(r1) > 0:
            if str(v.lemma_) in self.dt_list:
                r1a = v.text + ' ' + r1.text
            else:
                r1a = r1.text + ' ' + v.text
                r1a = self.check_pron(r1, v, ner_roles)

        if len(r2) > 0:
            if str(v.lemma_) in self.dt_list:
                r2a = v.text + ' ' + r2.text
            else:
                r2a = r2.text + ' ' + v.text
                r2a = self.check_pron(r2, v, ner_roles)

        if r1 and r2:
            ani_match = self.check_ani_match(r1, r2, ner_roles)

        sub_more_freq = self.get_freq(r1a, r2a)

        if 'each other' in sent.lower() or 'one another' in sent.lower():
            rcp_phrase = 1
            direction = 0
        else:
            rcp_phrase = 0

        features = {"trans": trans,
                    "trans_mod": trans_mod,
                    "v_tense": v_tense,
                    "v_act": v_act,
                    "modal": modal,
                    "neg": neg,
                    "is_root": is_root,
                    "direction": direction,
                    "sing_sub": sing_sub,
                    "sing_obj": sing_obj,
                    "conj_sub": conj_sub,
                    "conj_obj": conj_obj,
                    "rcp_phrase": rcp_phrase,
                    "ani_sub": ani_sub,
                    "ani_match": ani_match,
                    "sub_more_freq": sub_more_freq,
                    "num_np": num_np,
                    "num_clauses": num_clauses
                    }

        x_features = [trans, trans_mod, v_tense, v_act, modal, neg, is_root,
                      direction, sing_sub, sing_obj,
                      conj_sub, conj_obj, rcp_phrase, ani_sub, ani_match,
                      sub_more_freq, num_np, num_clauses]

        return features, x_features

    def encode(self, sentences, verbs):
        X = []
        err_sentences = []
        print("Encoding...")
        for k in range(len(sentences)):
            sentence = sentences[k]
            verb = verbs[k]
            try:
                print(sentence)
                features, x_features = self.get_features(sentence)
            except Exception as e:
                err_sentences.append(sentence)
                print("Parsing error:", sentence)
                print("error: ", e)
                features = {"trans": 'error',
                            "trans_mod": 'error',
                            "v_tense": 'error',
                            "v_act": 'error',
                            "modal": 'error',
                            "neg": 'error',
                            "is_root": 'error',
                            "direction": 'error',
                            "sing_sub": 'error',
                            "sing_obj": 'error',
                            "conj_sub": 'error',
                            "conj_obj": 'error',
                            "rcp_phrase": 'error',
                            "ani_sub": 'error',
                            "ani_match": 'error',
                            "sub_more_freq": 'error',
                            "num_np": 'error',
                            "num_clauses": 'error'
                            }

                x_features = ["trans", "trans_mod", "v_tense", "v_act", "modal", "neg",
                              "is_root",
                              "direction", "sing_sub", "sing_obj",
                              "conj_sub", "conj_obj", "rcp_phrase", "ani_sub", "ani_match",
                              "sub_more_freq", "num_np", "num_clauses"]

            X.append(x_features)

        return np.array(X), err_sentences


def main():
    import pandas as pd
    import time

    # test_sentences = ["Don Hornig married chemist Lilli Hornig.",
    #                "Angelina and Brad were married several years ago and had some issues."]
    # verbs = ['marry', 'marry']

    df = pd.read_csv('./data/sentences_feature_based_new.csv', sep=',', encoding='latin-1')
    test_sentences = df['sentence']
    verbs = df['pred']

    print("Starting...")
    encoder = FeatureBasedEncoder()
    start_time = time.time()
    fts, errors = encoder.encode(test_sentences, verbs)
    # for ft in fts:
    # print(ft)

    print("--- %s seconds ---" % (time.time() - start_time))
    col_name = 'error'
    df[col_name] = pd.Series(errors)
    df.to_csv('./data/sentences_feature_based_new_with_errors.csv', index=False)


if __name__ == '__main__':
    main()
