# Created by rahman at 21:35 2020-05-17 using PyCharm
"""
0. Install python >= 3.7
1. pip install requirements.txt
2. python -m nltk.downloader all
3. Paste the compressed patent dataset inside data dir
4. Download word embeddings from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit inside the model dir
"""


import sys
import gensim
import pandas as pd
from semantic_utils import *
from utils import *

#unzip_data()
#parse_xmls()

model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin.gz', binary=True)

df = pd.read_csv('./data' + "/df.csv", index_col=0, dtype=object)
prep_text(df, model)



### part 1
kw_search('Dow chemical company', df)
kw_search('cigarettes', df)


### part 2
df = pd.read_feather('{}{}.ftr'.format('./data/', 'df_tok_freq'))

semantic_search('petroleum', df, model)
semantic_search('tobacco', df, model)
semantic_search('cleaning', df, model)
semantic_search('cats', df, model)
semantic_search('happy life', df, model)

