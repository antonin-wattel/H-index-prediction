import json
from gensim.parsing.preprocessing import STOPWORDS

# store abstracts
def get_abstracts():
    paper_IDs = dict()
    with open('data/abstracts.txt') as f:
        for l in f:
            paper_ID, abstract = l.split("----",1)
            paper_IDs[int(paper_ID)] = json.loads(abstract)["InvertedIndex"].keys() - STOPWORDS
    return paper_IDs

# store authors
def get_authors():
    author_IDs = dict()
    with open('data/author_papers.txt') as f:
        for l in f:
            author_ID, papers = l.split(':')
            author_IDs[int(author_ID)] = map(int,papers.split('-'))
    return author_IDs