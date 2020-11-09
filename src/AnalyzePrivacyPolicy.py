#!/usr/local/bin/python
import re
import sys
import numpy as np

# Import DocSim
sys.path.append('/f20_cs858/src/')
from DocSim import DocSim


def compute_similarity(w2v_model, target_doc, target_sentence, input_num=10):
	ds = DocSim(w2v_model)
	data = list()
	# Read in the Privacy Policy
	with open(target_doc, 'r') as file:
		data_in = file.read()
		data.append(data_in)

	# Delimit by newline and period.
	#print("DATA: " + data)
	preliminary_tokens = re.split('\n|\\.', data[0])
	# Filter out empty sentences.
	tokens = list(filter(None,preliminary_tokens))
	results = ds.calculate_similarity(target_sentence, tokens)
	return results[:int(input_num)]