# Created by rahman at 21:41 2020-05-15 using PyCharm

import sys

import pandas as pd

from utils import keyword_search

# read dataframe
df = pd.read_csv('../data' + "/df.csv", index_col=0, dtype=object)

if __name__ == '__main__':

    words = sys.argv[1:] # space separated
    keyword_search(words, df)

