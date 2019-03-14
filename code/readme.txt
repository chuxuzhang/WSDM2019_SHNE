<1> Introduction 

code of SHNE in WSDM2019 paper: SHNE: Representation Learning for Semantic-Associated Heterogeneous Networks
Contact: Chuxu Zhang (czhang11@nd.edu)

<2> How to use

(install pytorch 1.0, de-compress word_embedding.txt.zip and het_random_walk_full.txt.zip) 

python SHNE.py [parameters]
(run with GPU: python SHNE.py --cuda 1)

#test academic dataset size: A_n - 28646, P_n - 21044, V_n - 18

<3> Data requirement
content.pkl: paper abstract content (paper_content, paper_content_id)
word_embedding.txt: pre-train word embedding of paper abstract
het_random_walk_full.txt: random/metapath walk as node sequences (corpus)


