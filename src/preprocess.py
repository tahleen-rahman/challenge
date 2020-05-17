# Created by rahman at 13:34 2020-05-17 using PyCharm
import gensim
import pandas as pd

from utils import parse_xmls, unzip_data

unzip_data(datapath='../data/')

parse_xmls(datapath='../data/')



from semantic_utils import prep_text

df = pd.read_feather('{}{}.ftr'.format('../data/', 'df_tok_freq'))
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin.gz', binary=True)

prep_text(df, model, datapath='../data/')
