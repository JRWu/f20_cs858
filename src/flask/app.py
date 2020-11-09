import glob
import os
from flask import Flask, render_template, request




# Import Model
# Will take about 10-15 seconds to load this model initially.
from gensim.models.keyedvectors import KeyedVectors
model_path = '/f20_cs858/models/GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Import Configuration for Analyzing Privacy Policy
import sys
sys.path.append('/f20_cs858/src/')
from AnalyzePrivacyPolicy import *



app = Flask(__name__, template_folder='/f20_cs858/templates/')

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == "POST":
		target_phrase = request.form['target']
		num_results = request.form['num']
		policy = request.form.get('comp_select')
		print("TARGET_POLICY: " + policy)
		semantic_similar_phrases = compute_similarity(w2v_model, policy, target_phrase, num_results)
		template = render_template('index.html',
			target_sequence = target_phrase,
			selected_policy = policy,
			policies = glob.glob('/f20_cs858/data/' + '*.txt'),
			results = semantic_similar_phrases)
		return template
	else:
		template = render_template('index.html',
		policies=glob.glob('/f20_cs858/data/' + '*.txt'))
		return template


if __name__ == '__main__':
    app.run()
