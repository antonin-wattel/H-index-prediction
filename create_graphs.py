import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
import scipy as sp

from load_data import *

    
author_IDs_to_paper_IDS = get_authors()
print('authors stored')

# extract information about authors and papers
paper_IDs_to_author_IDs = {}
for author, papers in author_IDs_to_paper_IDS.items():
    for paper in papers:
        paper_IDs_to_author_IDs.setdefault(paper, []).append(author)

G = nx.read_edgelist('data/coauthorship.edgelist', delimiter=' ', nodetype=int)

nodes = {k: v for v, k in enumerate(list(G.nodes()))}
inv_nodes = {v: k for k, v in nodes.items()}

authors = list(author_IDs_to_paper_IDS.keys())
papers = [p for ps in author_IDs_to_paper_IDS.values() for p in ps]

# Weighted collaboration graph
# create weighted collaboration graph
adj = nx.adjacency_matrix(G)
wadj = 0.5*adj.tolil()
for a in authors:
    ps = author_IDs_to_paper_IDS[a]
    for p in ps:
        auts = paper_IDs_to_author_IDs[p]
        for aut in auts:
            if a != aut:
                na = nodes[a]
                naut = nodes[aut]
                wadj[na, naut] += 1

WG = nx.from_scipy_sparse_matrix(wadj)
WG = nx.relabel_nodes(WG, inv_nodes)
nx.write_edgelist(WG, 'data/weighted_collaboration_network.edgelist', data=["weight"])


# Author similarity graph
# read embeddings of abstracts
text_embeddings = pd.read_csv("data/author_embeddings.csv", header=None)
text_embeddings = text_embeddings.rename(columns={0: "author_ID"})
text_embeddings = text_embeddings.fillna(0)

# create dictionnary of embeddings
embs = {}
for author in authors:
    embs[author] = text_embeddings[text_embeddings["author_ID"] == author]

# create author similarity graph
n = G.number_of_nodes()
simadj = sp.sparse.csr_matrix((n, n)).tolil()
for author1 in authors:
    papers = author_IDs_to_paper_IDS[author1]
    for paper in papers:
        coauthors = paper_IDs_to_author_IDs[paper]
        for author2 in coauthors:
            if author1 != author2:
                emb1 = embs[author1]
                emb2 = embs[author2]
                n_author1 = nodes[author1]
                n_author2 = nodes[author2]
                val = cosine(emb1, emb2)
                simadj[n_author1, n_author2] = val

SG = nx.from_scipy_sparse_matrix(simadj)
SG = nx.relabel_nodes(SG, inv_nodes)
nx.write_multiline_adjlist(SG, 'data/sim_collaboration_network.adjlist')