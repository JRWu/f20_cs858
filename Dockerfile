FROM python:3

RUN pip install pytextrank && \
	python -m spacy download en_core_web_sm

# Install dependencies for Word2Vec
RUN pip install numpy gensim

# Install dependencies for Flask Webapp
RUN pip install flask