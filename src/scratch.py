

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
# Loads slowly
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



import re
test_a = re.split('\n|\\.', all_policies_data[0])
pp_3 = list(filter(None, test_a))


test_b = re.split('\n|\\.', all_policies_data[1])
pp_1 = list(filter(None, test_b))

source_doc = pp_3
target_doc = pp_1
sim_scores = ds.calculate_similarity(source_doc[2], source_doc)


target_sentence = "We share your personal data with third parties."
sim_scores = ds.calculate_similarity(target_sentence, target_doc)





examples = list([
	"We share your personal data with third parties.",
	"You can delete your data at any time.",
	"We keep your data after you delete your profile."
])



sim_scores = ds.calculate_similarity(examples[2], pp_3)
sim_scores[:2]

sim_scores = ds.calculate_similarity(examples[1], pp_1)
sim_scores[:2]


# Deploy this?



import re
import sys
import numpy as np

# Import DocSim
sys.path.append('/f20_cs858/src/')
from DocSim import DocSim
from os import listdir
from os.path import isfile, join
import glob
import pytextrank
import spacy

from gensim.models.keyedvectors import KeyedVectors
model_path = '/f20_cs858/models/GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)


data_dir = "/f20_cs858/data/"
all_policies = glob.glob(data_dir + '*.txt')

target_doc = all_policies[0]
input_num = 10
target_sentence = "Share your your data with third parties."



#def compute_similarity(w2v_model, target_doc, target_sentence, input_num=10):
ds = DocSim(w2v_model)
data = list()
# Read in the Privacy Policy
with open(target_doc, 'r') as file:
	data_in = file.read()
	data.append(data_in)

# Delimit by newline and period.
preliminary_tokens = re.split('\n\n', data[0])
# Filter out empty sentences.
tokens = list(filter(None,preliminary_tokens))
results = ds.calculate_similarity(target_sentence, tokens)
results[:input_num]


results = compute_similarity(w2v_model, target_doc, target_sentence, input_num)
