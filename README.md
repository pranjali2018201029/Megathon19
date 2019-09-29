MEGATHON 19

QUALCOMM Problem Statement Solution

How to run :
python3 Doc_Retrieval.py "Doc_Filepath" "Abstract_Filepath" Summary_Size Abstract_Flag

here, 
Doc_Filepath : Absolute path to the csv file containing documents/articles \n
Abstract_Filepath : Absolute path to the csv file containing given query abstract \n
Summary_Size : Number of sentence in summary obtained from document content, further used to represent in a vector \n
                (Possible values: any integer less than no. of lines in a document) \n
Abstract_Flag : Optional flag to decide whether to use summary/abstract of document or use whole document without loss \n
                Default value is 1. (Possible values : 0 or 1) \n
                
Requirements/Dependencies : 
1. numpy
2. nltk
3. networkx
4. scipy
