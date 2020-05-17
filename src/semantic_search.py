# Created by rahman at 13:36 2020-05-17 using PyCharm
import sys
import gensim
import pandas as pd

from semantic_utils import sem_search


if __name__ == '__main__':

    df = pd.read_feather('{}{}.ftr'.format('../data/', 'df_tok_freq'))
    model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin.gz', binary=True)

    wordlist = sys.argv[1:]
    sem_search(wordlist, df, model)
