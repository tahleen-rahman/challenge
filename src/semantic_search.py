# Created by rahman at 13:36 2020-05-17 using PyCharm
import sys
import gensim
import pandas as pd

from semantic_utils import sem_search


if __name__ == '__main__':

    df1 = pd.read_csv('../data' + "/df.csv", index_col=0, dtype=object)

    df = pd.read_feather('{}{}.ftr'.format('../data/', 'df_tok_freq'))
    model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin.gz', binary=True)

    wordlist = sys.argv[1:]
    sem_search(wordlist, df, df1, model, 5, 3)
