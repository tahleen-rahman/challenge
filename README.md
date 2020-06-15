Semantic and keyword search in a dataset of patent documents using googleNews word2vec embeddings 
Usage from terminal

0. Install python >= 3.7
1. pip3 install -r requirements.txt
2. python -m nltk.downloader all
3. Paste the compressed patent dataset inside data dir
4. Download word embeddings from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit inside the model dir
5. python3 preprocess.py
6. python3 keyword_search.py keyword1 keyword2 keywordN
7. python3 semantic_search word1 word2 (Max 2 words together supported right now)

Or follow steps in console_steps.py in a python console
