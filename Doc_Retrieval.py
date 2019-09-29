import csv
import math
import numpy as np
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

import porter as pt


## Global Variable to store inverted index
Inv_Index = {}
Query_Inv_Index = {}

## List of Doc Abstract objects
Doc_Abstract_List = []

## List of Query Abstract tokens list
Query_Abstract_List = []

Doc_Id = 0

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


def textRank(df, docNum, abstractLen):

    #Sentence tokenizer
    s = str(df[docNum])
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

    abstract = ""
    i = 0
    for s in rankedSentences:
        if i < abstractLen:
# abstract.append(rankedSentences[i][1])
            abstract += rankedSentences[i][1] + " "
        i += 1
    return abstract


class Abstract:
    def __init__(self, text="", token=[]):
        self.ID = Doc_Id
        self.text = text
        self.token = token


def Pre_Processing(Abstract):
    
    # case folding
    Abstract = Abstract.lower()

    #split into words
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    words_tokens = tokenizer.tokenize(Abstract)

    #stop word removal
    stop_words = set(stopwords.words('english'))
    words_tokens = [token for token in words_tokens if token not in stop_words]

    #stemming
    words_tokens = [pt.stem(token) for token in words_tokens]

    return words_tokens


def Create_Abstract(csv_filename, Abstract_Size):
    
    global Doc_Abstract_List
    global Doc_Id
    
    df = []
    with open(csv_filename,'r')as f:
        data = csv.reader(f)
        
        for DocData in data:
            df.append(DocData[0])
            
        for i in range(len(df)):
            Abstract_str = textRank(df, i, Abstract_Size)
            print("DOCUMENT")
            print(df[i])
            print("ABSTRACT")
            print(Abstract_str)
            abstract_tokens = Pre_Processing(Abstract_str)
            Abstract_Obj = Abstract(Abstract_str, abstract_tokens)
            Doc_Abstract_List.append(Abstract_Obj)
            Doc_Id += 1


def Read_Abstract_file(csv_filename):
    global Doc_Abstract_List
    global Doc_Id
    
    with open(csv_filename,'r')as f:
        data = csv.reader(f)
        
        for Doc_Data in data:
            
            Doc_Abstract = textRank
            abstract_tokens = Pre_Processing(abstract_row[0])
            Abstract_Obj = Abstract(abstract_row, abstract_tokens)
            Doc_Abstract_List.append(Abstract_Obj)
            Doc_Id += 1


def Read_Query_Abstract(csv_filename):
    global Query_Abstract_List
    with open(csv_filename,'r')as f:
        data = csv.reader(f)
        
        for abstract_row in data:
            abstract_tokens = Pre_Processing(abstract_row[0])
            Query_Abstract_List.append(abstract_tokens)


# In[35]:


## Create Index for Document Abstract

def Create_BOW():
    
    global Inv_Index
    
    for Doc_Abstract_Obj in Doc_Abstract_List:
        for token in Doc_Abstract_Obj.token:
            if token in Inv_Index:
                if Doc_Abstract_Obj.ID in Inv_Index[token].keys():
                    Inv_Index[token][Doc_Abstract_Obj.ID] += 1
                else:
                    Inv_Index[token][Doc_Abstract_Obj.ID] = 1
            else:
                Inv_Index[token] = {}
                Inv_Index[token][Doc_Abstract_Obj.ID] = 1
            
def TF_IDF():
    
    global Inv_Index
    N = len(Doc_Abstract_List)
    
    for term in Inv_Index.keys():
        df = len(Inv_Index[term])
        idf = 0
        if(df>0):
            idf = math.log(N/df)
            
        for DocID in Inv_Index[term].keys():
            tf = Inv_Index[term][DocID]   
            Inv_Index[term][DocID] = tf*idf
            

## Get Posting Lists for query words and
## Tranform Posting lists of words in query abstract into 
## Vector representationsn (TF_IDF score) of document abstracts

def Docs_Vec_Transform():
    
    l = len(Query_Inv_Index.keys())
    Doc_Term_Matrix = {}
    i = 0
    
    for token in Query_Inv_Index.keys():
        if token in Inv_Index.keys():
            Posting_List = Inv_Index[token]

            for DocID in Posting_List.keys():

                if DocID not in Doc_Term_Matrix:
                    Doc_Term_Matrix[DocID] = [0]*l

                Doc_Term_Matrix[DocID][i] = Posting_List[DocID]
        i += 1
        
    return Doc_Term_Matrix


## Calculate TF-IDF score of Query Abstract, Vector Representation of Query Abstract

def Query_Vec_Transform(Query_Tokens):
        
    global Query_Inv_Index
    
    for token in Query_Tokens:
        if token not in Query_Inv_Index:
            Query_Inv_Index[token] = 1
        else:
            Query_Inv_Index[token] +=1
    
    N = len(Doc_Abstract_List)
    i = 0
    Query_Arr = [0]*len(Query_Inv_Index.keys())
    
    for term in Query_Inv_Index.keys(): 
        tf = Query_Inv_Index[term]
        
        df = 0
        if term in Inv_Index.keys():
            df = len(Inv_Index[term])
            
        idf = 0
        if(df>0):
            idf = math.log(N/df)
        Query_Arr[i] = tf*idf
        i += 1
        
    return Query_Arr


## Cosine Similarity fuction : I/P - Two vectors, O/P - Similarity value

def Cosine_Similarity(Query_Vec, Doc_Vec):
    
    Query_Arr = np.array(Query_Vec)
    Doc_Arr = np.array(Doc_Vec)
    
    Dot_Product = np.dot(Query_Arr, Doc_Arr)
    
    Cosine_Sim = Dot_Product / (np.linalg.norm(Query_Arr)*np.linalg.norm(Doc_Arr))

    return Cosine_Sim


## Create Similarity Matrix

def Create_Similarity_Matrix(Queryfile, Docfile, Abstract_Size):
    
    Create_Abstract(Docfile, Abstract_Size)
    Read_Query_Abstract(Queryfile)
    
    Create_BOW()
    TF_IDF()
    
    No_Queries = len(Query_Abstract_List)
    No_Docs = len(Doc_Abstract_List)
    
    Similarity_mat = np.zeros((No_Queries, No_Docs))
    
    for i in range(No_Queries):
        
        Query_Tokens = Query_Abstract_List[i]
        Query_Vec = Query_Vec_Transform(Query_Tokens)
        Doc_Term_Matrix = Docs_Vec_Transform()
        
        for DocID in Doc_Term_Matrix.keys():
            Doc_Vec = Doc_Term_Matrix[DocID]
            Cosine_Sim = Cosine_Similarity(Query_Vec, Doc_Vec)
            Similarity_mat[i][DocID] = Cosine_Sim
            
    return Similarity_mat


Similarity_mat = Create_Similarity_Matrix("./abstract.csv", "./body_text.csv", 5)
print(Similarity_mat)
