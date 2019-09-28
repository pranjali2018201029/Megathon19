import csv
import math
import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import porter

## Global Variable to store inverted index
Inv_Index = {}
Query_Inv_Index = {}

## List of Doc Abstract objects
Doc_Abstract_List = []

## List of Query Abstract tokens list
Query_Abstract_List = []

Doc_Id = 0


class Abstract:
    def __init__(self, text="", token=[]):
        self.ID = Doc_Id
        self.text = text
        self.token = token

def Pre_Processing(Abstract):
    
    # case folding
    Abstract = Abstract.tolower()

    #split into words
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    words_tokens = tokenizer.tokenize(Abstract)

    #stop word removal
    stop_words = set(stopwords.words('english'))
    words_tokens = [token for token in words_tokens if token not in stop_words]

    #stemming
    stemmed_tokens = porter(words_tokens)

    return stemmed_tokens

def Read_Abstract_file(csv_filename):
    global Doc_Abstract_List
    global Doc_Id
    
    with open(csv_filename,'r')as f:
        data = csv.reader(f)
        
        for abstract_row in data:
            Doc_Id += 1
            abstract_tokens = Pre_Processing(abstract_row)
            Abstract_Obj = Abstract(abstract_row, abstract_tokens)
            Doc_Abstract_List.append(Abstract_Obj)

def Read_Query_Abstract(csv_filename):
    global Query_Abstract_List
    with open(csv_filename,'r')as f:
        data = csv.reader(f)
        
        for abstract_row in data:
            abstract_tokens = Pre_Processing(abstract_row)
            Query_Abstract_List.append(abstract_tokens)


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
    N = len(Doc_Astract_List)
    
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
    
    Doc_Term_Matrix = {}
    l = len(Query_Tokens)
    i = 0
    for token in Query_Inv_Index.keys():
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
    Cosine_Sim = Dot_Product / (np.linalg.norm(Query_Arr)+np.linalg.norm(Doc_Arr))
    
    return Cosine_Sim


## Create Similarity Matrix

def Create_Similarity_Matrix(Queryfile, Docfile):
    
    Read_Abstract_file(Docfile)
    Read_Query_Abstract(Queryfile)
    
    Create_BOW()
    TF_IDF()
    
    No_Queries = len(Query_Abstract_List)
    No_Docs = len(Doc_Abstract_List)
    
    Similarity_mat = np.zeroes((No_Queries, No_Docs))
    
    for i in range(No_Queries):
        
        Query_Tokens = Query_Abstract_List[i]
        Query_Vec = Query_Vec_Transform(Query_Tokens)
        Doc_Term_Matrix = Docs_Vec_Transform()
        
        for j in range(len(Doc_Term_Matrix)):
            Doc_Vec = Doc_Term_Matrix[j]
            Cosine_Sim = Cosine_Similarity(Query_Vec, Doc_Vec)
            Similarity_mat[i][j] = Cosine_Sim
    return Similarity_mat


Similarity_mat = Create_Similarity_Matrix("", "")
print(Similarity_mat)

