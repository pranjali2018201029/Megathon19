import numpy as np
import csv
import nltk
nltk.download('punkt') # one time execution
import re
#Imported to tokenize text into sentences
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import porter
import networkx as nx
import scipy
from scipy.sparse import csr_matrix

stopWords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[a-zA-Z0-9]+')



def pagerank(G, alpha=0.85, personalization=None, 
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight', 
             dangling=None): 
    if len(G) == 0: 
        return {} 
  
    if not G.is_directed(): 
        D = G.to_directed() 
    else: 
        D = G 
  
    # Create a copy in (right) stochastic form 
    W = nx.stochastic_graph(D, weight=weight) 
    N = W.number_of_nodes() 
  
    # Choose fixed starting vector if not given 
    if nstart is None: 
        x = dict.fromkeys(W, 1.0 / N) 
    else: 
        # Normalized nstart vector 
        s = float(sum(nstart.values())) 
        x = dict((k, v / s) for k, v in nstart.items()) 
  
    if personalization is None: 
  
        # Assign uniform personalization vector if not given 
        p = dict.fromkeys(W, 1.0 / N) 
    else: 
        missing = set(G) - set(personalization) 
        if missing: 
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing) 
        s = float(sum(personalization.values())) 
        p = dict((k, v / s) for k, v in personalization.items()) 
  
    if dangling is None: 
  
        # Use personalization vector if dangling vector not specified 
        dangling_weights = p 
    else: 
        missing = set(G) - set(dangling) 
        if missing: 
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing) 
        s = float(sum(dangling.values())) 
        dangling_weights = dict((k, v/s) for k, v in dangling.items()) 
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0] 
  
    # power iteration: make up to max_iter iterations 
    for _ in range(max_iter): 
        xlast = x 
        x = dict.fromkeys(xlast.keys(), 0) 
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes) 
        for n in x: 
  
            # this matrix multiply looks odd because it is 
            # doing a left multiply x^T=xlast^T*W 
            for nbr in W[n]: 
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight] 
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] 
  
        # check convergence, l1 norm 
        err = sum([abs(x[n] - xlast[n]) for n in x]) 
        if err < N*tol: 
            return x 
    return x
    # raise NetworkXError('pagerank: power iteration failed to converge '
                        # 'in %d iterations.' % max_iter) 


def textRank(docNum, abstractLen):
	
	#Read document
	df = []
	with open("body_text.csv", 'r') as f:
		data = csv.reader(f)
		for row in data:
			df.append(row[0])

	#Sentence tokenizer
	s = str(df[docNum-1])
	print("Document:")
	print(s)
	print("\n")
	sentences = sent_tokenize(str(s))
	bagOfWords = {}
	i = 0

	#For each sentence, clean
	for sentence in sentences:
		tokens = tokenizer.tokenize(sentence)
		finalTokens = {}
		for token in tokens:
			token = token.casefold()
			#Bag of Words
			if token not in stopWords and (token.isalpha() or token.isnumeric()):
				token = porter.stem(token)
				finalTokens[token] = 1
		if len(finalTokens) > 0:
			bagOfWords[i] = finalTokens
		i += 1

	#Matrix N*N (N = no. of sentences)
	#Jaccard similarity for each sentence with each sentence (i,j)
	lenSecSent = 0
	i = 0
	j = 0
	similarityMatrix = np.zeros((len(sentences), len(sentences)))
	for sentence in bagOfWords:
		BOW1 = bagOfWords[sentence]
		for secondSentence in bagOfWords:
			if sentence != secondSentence:
				BOW2 = bagOfWords[secondSentence]
				lenSecSent = len(BOW2)
				j = 0
				for token in BOW1:
					if token in BOW2:
						j += 1
			similarityMatrix[sentence][secondSentence] = j/(len(BOW1) + lenSecSent - j)
			similarityMatrix[secondSentence][sentence] = j/(len(BOW1) + lenSecSent - j)

	#Page rank function
	similarityMatrix = scipy.sparse.csr_matrix(similarityMatrix)
	nx_graph = nx.from_scipy_sparse_matrix(similarityMatrix)
	scores = pagerank(nx_graph, max_iter=1000)

	#Sorting for top documents
	rankedSentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

	abstract = []
	i = 0
	for s in rankedSentences:
		if i < abstractLen:
			abstract.append(rankedSentences[i][1])
		i += 1
	return abstract


abstract = textRank(4, 4)
print("\n\n")
print("Abstract:")
print(abstract)
