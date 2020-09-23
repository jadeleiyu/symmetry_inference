import numpy as np
import nltk
import codecs
import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import os
import shutil
import hashlib
from sys import platform


def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for line in f:
        count += 1
    return count


def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 300)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    return model


def tokenize_text(sentences):
    sents = []
    for sent in sentences:
        sent = nltk.sent_tokenize(sent)
        tokens = []
        for subsent in sent:
            for word in nltk.word_tokenize(subsent):
                tokens.append(word.lower())

        sents.append(" ".join(tokens))

    return sents


def apply_weight(sents):
    word_idf_weight = []
    tfidf = TfidfVectorizer()
    tfidf.fit(sents)
    max_idf = max(tfidf.idf_)
    word_idf_weight = defaultdict(lambda: max_idf, [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
    return word_idf_weight


def get_avg_vector(model, words, word_idf_weight):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in model.vocab:
            mean.append(model.syn0[model.vocab[word].index] * word_idf_weight[word])
            all_words.add(model.vocab[word].index)

    if not mean:
        return np.zeros(model.layer1_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_avg_list(model, sentences):
    td = apply_weight(sentences)
    return np.vstack([get_avg_vector(model, s, td) for s in sentences])