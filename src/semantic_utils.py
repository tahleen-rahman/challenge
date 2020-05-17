# Created by rahman at 13:36 2020-05-17 using PyCharm
import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from collections import Counter

from utils import format_results, colored_results



def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def prep_text(df, model, datapath='./data/', stemming=0):
    """
    0. replace NA/missing values with ""
    1. filter the non english patents for now,
    2. remove punctuations,
    3. stop words
    4. lemmatize
    5. stemming?
    and what else?

    :return:
    """
    print("Embeddings loaded, Preparing data for semantic analysis...")

    #0. replace NA/missing values with ""
    df.fillna('', inplace=True)


    #1. filter the non english patents for now,
    print ("No Multiligual support for semantic search yet, Please give English inputs only.")
    df = df[df.lang=='en']
    print (df.shape[0], "english patents exist.")

    # combine all text
    df['text'] = df[['titles', 'abstract', 'descriptions', 'claims']].values.tolist()


    # 2. remove punctuations,
    print("Removing punctuations")

    df['text'] = df.text.apply(lambda text: str(text))
    df['text'] = df.text.apply(word_tokenize)
    df['text'] = df.text.apply(lambda text: [word for word in text if word.isalpha()])


    """df['title_tok'] = df.titles.apply(word_tokenize)
    df['title_tok'] = df.title_tok.apply(lambda text: [word for word in text if word.isalpha()])

    df['abstract_tok'] = df['abstract'].apply(word_tokenize)
    df['abstract_tok'] = df.abstract_tok.apply(lambda text: [word for word in text if word.isalpha()])

    df['desc_tok'] = df['descriptions'].apply(word_tokenize)
    df['desc_tok'] = df.desc_tok.apply(lambda text: [word for word in text if word.isalpha()])

    df['claim_tok'] = df['claims'].apply(word_tokenize)
    df['claim_tok'] = df.claim_tok.apply(lambda text: [word for word in text if word.isalpha()])"""


    # filter out stop words from all languages
    print("Removing stopwords...")

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    stop_words_de = set(stopwords.words('german'))
    stop_words.update(stop_words_de)
    stop_words.update(stop_words_fr)

    df['text'] = df.text.apply(lambda text: [w for w in text if not w in stop_words])



    """
    df['title_tok'] = df.title_tok.apply(lambda text: [w for w in text if not w in stop_words])
    df['abstract_tok'] = df.abstract_tok.apply(lambda text: [w for w in text if not w in stop_words])
    df['desc_tok'] = df.desc_tok.apply(lambda text: [w for w in text if not w in stop_words])
    df['claim_tok'] = df.claim_tok.apply(lambda text: [w for w in text if not w in stop_words])"""



    # converts the word to its meaningful base form, infer the POS automatically
    print("Lemmatizing...")

    lemmatizer = WordNetLemmatizer()
    df['text'] = df.text.apply(lambda text: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text])


    """
    df['title_tok'] = df.title_tok.apply(lambda text: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text])
    df['abstract_tok'] = df.abstract_tok.apply(lambda text: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text])
    df['desc_tok'] = df.desc_tok.apply(lambda text: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text])
    df['claim_tok'] = df.claim_tok.apply(lambda text: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text])"""

    # stemming of words
    if stemming:
        porter = PorterStemmer()
        df['text'] = df.text.apply(lambda text: [porter.stem(word) for word in text])

        """
        df['title_tok'] = df.title_tok.apply(lambda text: [porter.stem(word) for word in text])
        df['abstract_tok'] = df.abstract_tok.apply(lambda text:[porter.stem(word) for word in text])
        df['desc_tok'] = df.desc_tok.apply(lambda text:[porter.stem(word) for word in text])
        df['claim_tok'] = df.claim_tok.apply(lambda text: [porter.stem(word) for word in text])"""



    # load model and filter words that exist in our pretrained word2vec model
    print("Infering semantics")

    df['text'] = df.text.apply(lambda text: [word for word in text if word in model.vocab])
    df.dropna(inplace=True)
    df = df[df.text.apply(lambda text: len(text)>0)]

    print("Counting occurences")
    # count frequency of words
    df['freq_dict'] = df.text.apply(lambda text: dict(Counter(text)))


    # save in ftr to disk
    df = df.reset_index(drop=True)
    df.to_feather(datapath + 'df_tok_freq.ftr')

    print ("Dataframe saved to ", datapath + "df_tok_freq.ftr  with columms", df.columns.to_list())



def root_word_search_rank(synonym, simi, df):
    """

    :param synonym:
    :param simi: simlarity score of synonym
    :return:
    """

    res = df
    res = res[res.text.apply(lambda text: synonym in text)]
    res_ = res.copy()

    #rank results
    if len(res) > 0:

        res_['score'] = res.freq_dict.apply(lambda x: x[synonym] * simi)
        res = res_.sort_values(by='score', ascending=False)

        format_results(res, synonym, ranked=1)
        colored_results(res, synonym,  flag_phrase=1, ranked=1)

        return 1

    else:
        print ("Nothing for", synonym)
        return 0


def sem_search(wordlist, df, model):
    """

    :return:
    """

    #phrase = ' '.join(wordlist)

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    wordlist = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in wordlist]

    # only words that exist in the word2vec model
    wordlist = [word for word in wordlist if word in model.vocab]


    for word in wordlist:

        # Search for the same word root
        root_word_search_rank(synonym=word, simi=1.0, df=df)

        syn_dict = dict(model.most_similar(word))

        for synonym, simi in syn_dict.items():
            #print("Searching for", synonym, "similarity", simi)
            root_word_search_rank(synonym, simi, df)



def semantic_search(input, df, model):

    # split by whitespace(s) into list
    wordlist = input.split()
    sem_search(wordlist, df, model)
