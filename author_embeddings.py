import gensim.downloader as api
import numpy as np
from load_data import *
import re
from gensim.parsing.preprocessing import STOPWORDS

#try with different pre-trained word-embedding models
wv = api.load('word2vec-google-news-300')
#wv = api.load('glove-wiki-gigaword-300')
print("Word2Vec model loaded")
n = wv.vector_size

paper_IDs = get_abstracts()
print('abstracts stored')
    
author_IDs = get_authors()
print('authors stored')


# get paper value from paper_ID
def get_paper_value(paper_ID):
    vec = np.zeros(n)
    try:
        words_used = set()
        for token in paper_IDs[paper_ID]:
            words = re.sub(r'[-/]', ' ', re.sub(r'[.…,:?!;\'‘’"“”()*–]|[0-9]+-|[0-9]|\'s', '', token))
            for w in words.split():
                if w not in STOPWORDS and w not in words_used:
                    words_used.add(w)
                    try:
                        vec += wv[w]
                    except:
                        continue
    except:
        pass
    return vec

def get_author_value(author_ID):
    vec = np.zeros(n)
    for paper_ID in author_IDs[author_ID]:
        vec += get_paper_value(paper_ID)
    return vec



df = open("data/author_embeddings.csv", "w")

for author_ID in author_IDs:
    v = get_author_value(author_ID)
    df.write(str(author_ID)+ "," + ",".join(map(lambda x: "{:.8f}".format(round(x, 8)), v))+"\n")

df.close()