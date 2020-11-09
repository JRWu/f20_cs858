

pip install numpy==1.16.2
pip install gensim==3.8.1



pip install numpy gensim




cd /f20_cs858/models

# Obtain the model
# https://drive.google.com/u/0/uc?export=download&confirm=DdnH&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

# Unzip the model
gzip -d GoogleNews-vectors-negative300.bin.gz






from os import listdir
from os.path import isfile, join
import glob

import pytextrank
import spacy

data_dir = "/f20_cs858/data/"
all_policies = glob.glob(data_dir + '*.txt')



all_policies_data = list()
for pol in all_policies:
	print(pol)
	with open(pol, 'r') as file:
		data_in = file.read()
		all_policies_data.append(data_in)


# Word2Vec data
from gensim.models.keyedvectors import KeyedVectors
model_path = '/f20_cs858/models/GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Append 
import sys
sys.path.append('/f20_cs858/src/')
from DocSim import DocSim
ds = DocSim(w2v_model)

import numpy as np


source_doc = all_policies_data[0]
target_docs = list(all_policies_data[1:])
sim_scores = ds.calculate_similarity(source_doc, target_docs)

for s in sim_scores:
	print(s['score'])


# IDEA: Tokenize each document by sentence.
# For each document, compare it to each other document's sentences
# Keep the 80-90% most common phrases

# IDEA: Full document similarity computation


