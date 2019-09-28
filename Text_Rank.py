import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re
#Imported to tokenize text into sentences
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import porter
import networkx as nx

stopWords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[a-zA-Z0-9]+')

def textRank(docNum, abstractLen):
	
	#Read document
	df = pd.read_csv("tennis_articles_v4.csv")

	#Sentence tokenizer
	s = df[docNum]
	sentences = sent_tokenize(s)

	bagOfWords = {}
	i = 0

	#For each sentence, clean
	for sentence in sentences:
		tokens = tokenizer.tokenize(sentence)
		for token in tokens:
			token = token.casefold()
			#Bag of Words
			finalTokens = {}
			if token not in stopWords and (token.isalpha() or token.isnumeric()):
				token = porter.stem(token)
				finalTokens[token] = 1
		bagOfWords = {i : finalTokens}
		i += 1

	#Matrix N*N (N = no. of sentences)
	#Jaccard similarity for each sentence with each sentence (i,j)
	similarityMatrix = np.zeros((len(sentences), len(sentences)))
	for sentence in bagOfWords:
		BOW1 = bagOfWords[sentence]
		for secondSentence in bagOfWords:
			if sentence != secondSentence:
				BOW2 = bagOfWords[secondSentence]
				i = 0
				for token in BOW1:
					if token in BOW2:
						i += 1
			similarityMatrix[sentence][secondSentence] = i/(len(BOW2) + len(BOW2) - i)
			similarityMatrix[secondSentence][sentence] = i/(len(BOW2) + len(BOW2) - i)

	#Page rank function
	nx_graph = nx.from_scipy_sparse_matrix(similarityMatrix)
	scores = nx.pagerank(nx_graph)

	#Sorting for top documents
	rankedSentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
	abstract = []
	i = 0
	for s in rankedSentences:
		if i < abstractLen:
			abstract.append(rankedSentences[i][1])
		i += 1
	return abstract


