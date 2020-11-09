#!/usr/local/bin/python
"""
NAIVE text ranking. Explore a document at the phrase level.

"""

from os import listdir
from os.path import isfile, join
import glob

import pytextrank
import spacy

data_dir = "/f20_cs858/data/"
#all_policies = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

all_policies = glob.glob(data_dir + '*.txt')


for pol in all_policies:
	print(pol)



p_test = all_policies[0]











pp_15 = "15_sleep_science_alarm_clock.txt"

with open(data_dir + pp_15, 'r') as file:
	data_pp_15 = file.read()


# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

doc = nlp(data_pp_15)

# Examine the top-ranked phrases in the document
for p in doc._.phrases:
    print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
    print(p.chunks)


