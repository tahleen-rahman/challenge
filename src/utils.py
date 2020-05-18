# Created by rahman at 01:15 2020-05-16 using PyCharm

import re
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as et
import zipfile, fnmatch, os
from colorama import Fore, Back
from colorama import init
init(autoreset=True)



def unzip_data(datapath='./data/'):

    # Unzip all
    for root, dirs, files in os.walk(datapath):

        for filename in fnmatch.filter(files, '*.zip'):

            if os.path.exists(os.path.join(root, os.path.splitext(filename)[0])):
                print(os.path.join(root, os.path.splitext(filename)[0]), " exists")

            else:
                print("extracting", os.path.join(root, filename))
                zipfile.ZipFile(os.path.join(root, filename)).extractall(os.path.join(root, os.path.splitext(filename)[0]))


    # remove all except xml and zip files
    for root, dirs, files in os.walk(datapath):

        for filename in files:

            if (filename[-3:] not in ['xml', 'zip', 'pdf']):
                os.remove(os.path.join(root, filename))




def parse_xml(root, xml_file):
    """
    parse all the useful fields and store into dataframe
    :param xml_file:
    :return:
    """

    xtree = et.parse(os.path.join(root, xml_file))
    xroot = xtree.getroot()

    try:
        lang = xroot.get('lang')
    except:
        lang = ''

    try:
        pub_date = str(xroot.get('date-publ'))
    except:
        pub_date = ''

    try:
        pub_num = str(xroot.get('doc-number'))
    except:
        pub_num = ''

    try:
        abstract = xroot.findall("./abstract/p")[0].text
    except:
        abstract = ''

    try:
        applicants = [item.text for item in xroot.findall("./SDOBI/B700/B710/B711/snm")]
    except:
        applicants = ''

    try:
        inventors = [item.text for item in xroot.findall("./SDOBI/B700/B720/B721/snm")]
    except:
        inventors = ''

    try:
        titles = [item.text for item in xroot.findall("./SDOBI/B500/B540/B542")]  # use for semantics
    except:
        titles = ''

    try:
        descriptions = [item.text for item in xroot.findall("./description/p")]  # use for semantics
    except:
        descriptions = ''

    try:
        claims = [item.text for item in xroot.findall("./claims/claim/claim-text")]  # use for semantics
    except:
        claims = ''

    try:
        ipc_num = [item.text.replace(' ','') for item in xroot.findall("./SDOBI/B500/B510/")[1:]]  # the first number is always 3, so ignore it now
    except:
        ipc_num = ''

    return xml_file[:-4], ipc_num, pub_num, lang, pub_date, applicants, inventors, titles,  abstract, descriptions, claims




def parse_xmls(datapath='./data/'):
    """
    read xml files and load data into dataframe, clean missing values and save in df.csv
    :return:
    """

    arr = []

    for root, dirs, files in os.walk(datapath):

        for filename in files:

            if (filename[-3:] == 'xml'):
                if (filename != 'TOC.xml'):
                    arr.append(parse_xml(root, filename))


    df = pd.DataFrame(data = arr, columns=['fname', 'ipc_num', 'pub_num', 'lang', 'pub_date', 'applicants', 'inventors', 'titles', 'abstract', 'descriptions', 'claims'])

    print(df.shape[0], 'patent records read.', df.shape[0] - df.dropna().shape[0], 'rows have some missing values. Saving to dataframe to df.csv')

    #df.dropna(inplace=True)

    df.to_csv(datapath + "/df.csv")

    return df



def format_results(res, words, ranked, topn=5):
    """
    tune and display
    :return:
    """
    pd.options.display.max_rows=topn
    #pd.options.display.max_cols=10
    pd.options.display.max_colwidth=30
    pd.options.display.expand_frame_repr=0
    #pd.options.display.width = 500
    print ("")
    print ("")
    print ("")

    print (Back.GREEN + str(len(res))+ " patents  containing", Fore.RED + str(words), Back.GREEN + "found")

    cols = ['score', 'fname', 'ipc_num', 'pub_num', 'pub_date',
            'applicants', 'inventors', 'titles', 'abstract',
            'descriptions', 'claims']

    if not ranked:
        cols.pop(0)

    print (res[cols])
    #res.fname.to_csv(datapath + "/result.csv")

    return len(res)


def colored_results(res, wordlist, flag_phrase, ranked=0, topn=5):
    """
    color the exact matches in red
    :param res:
    :param wordlist:
    :return:
    """

    #print (wordlist)
    if flag_phrase:
        phrase = wordlist
        my_regex = r"(" + re.escape(phrase.lower()) + r")"
    else:
        my_regex = r"\b("
        for word in wordlist:
            my_regex = my_regex + re.escape(word.lower()) + "|"
        my_regex += r")\b"

    #print (my_regex)

    #my_regex = r'\b(?:{})\b'.format('|'.join(map(re.escape, wordlist)))
    #print (my_regex)

    cols = ['score', 'fname', 'ipc_num', 'pub_num', 'pub_date',
            'applicants', 'inventors', 'titles', 'abstract']
        #,'descriptions', 'claims']

    if not ranked:
        cols.pop(0)

    print ("\n")
    if ranked:
        print("Top", topn, "results:")

    res = res[cols]
    #res.loc[:,'score'] = res.loc[:,'pub_date'].copy(deep=True).astype(str)

    for row in res[cols].iloc[:topn].values:

        print("")

        for i in range(len(row)):
            try:
                print(res.columns[i].upper() + ": "+ re.sub(my_regex, Fore.RED + r'\1' + Fore.RESET, str(row[i].lower()), flags=re.IGNORECASE))
            except:
                print (" ")



def parse_dates(wordlist):
    """
    unifies multiple date formats if present
    :param wordlist:
    :return:
    """
    for i in range(len(wordlist)):

        for format in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y%m%d'):

            try:
                date_obj = datetime.strptime(wordlist[i], format).date()
                print("date detected, ", wordlist[i])
                wordlist[i] = date_obj.strftime('%Y%m%d')
                break

            except ValueError:
                pass

    return wordlist



def keyword_search(wordlist, df, topn):
    """

    :param wordlist: list contaiing keywords
    :return:
    """


    # dates
    wordlist = parse_dates(wordlist)

    # initialize result df
    res = df

    # find exact ordered match first among key fields
    phrase = ' '.join(wordlist)
    res = res[res[['fname', 'ipc_num', 'pub_num', 'lang','pub_date',
            'applicants', 'inventors', 'titles']].apply(lambda row: row.astype(str).str.contains(phrase, case=False).any(), axis=1)]

    # display
    if len(res) > 0:
        format_results(res, phrase, ranked=0, topn=topn)
        colored_results(res, phrase, flag_phrase=1, ranked=0, topn=topn)


    else:

        print("No patents found containing phrase", phrase , " in key fields, Searching as separate keywords...")

        res = df
        #iteratively search for records containing all keywords unordered in key fields
        for word in wordlist:
            word_regex = r'(\b' + re.escape(word.lower()) + r'\b)'
            res = res[res[['fname', 'ipc_num', 'pub_num', 'lang','pub_date',
            'applicants', 'inventors', 'titles']].apply(lambda row: row.astype(str).str.contains(word_regex, case=False).any(), axis=1)]

        # display
        if len(res) > 0:
            format_results(res, wordlist, ranked=0, topn=topn)
            colored_results(res, wordlist, flag_phrase=0, ranked=0, topn=topn)


        else:

            print("No patents found containing all words", wordlist, "in key fields. Searching as phrase in text ...")

            res = df
            res = res[res[['abstract', 'descriptions', 'claims']].apply(lambda row: row.astype(str).str.contains(phrase, case=False).any(), axis=1)]

            # display
            if len(res) > 0:
                format_results(res, phrase, ranked=0, topn=topn)
                colored_results(res, phrase, flag_phrase=1, ranked=0, topn=topn)

            else:

                print("No patents found containing phrase", phrase, "in text. Searching as separate keywords in text ...")

                res = df
                # iteratively search for records containing all keywords unordered in text
                for word in wordlist:
                    word_regex = r'(\b' + re.escape(word.lower()) + r'\b)'
                    res = res[res.apply(lambda row: row.astype(str).str.contains(word_regex, case=False).any(), axis=1)]

                # display
                if len(res) > 0:
                    format_results(res, wordlist, ranked=0, topn=topn)
                    colored_results(res, wordlist, flag_phrase=0, ranked=0, topn=topn)

                else:

                    print("No patents found containing all words", wordlist, "in text. Try removing a keyword or try again with another term.")


def kw_search(input, df, topn=5):

    # split by whitespace(s) into list
    words = input.split()
    keyword_search(words, df, topn)
