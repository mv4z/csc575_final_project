#!/usr/bin/python

#imports
import os
import pandas as pd
import numpy as np
import typing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

def parse_documents_file(project_directory: str, save_documents=False):
    """takes the source data’s directory as input, parses through the document data, and returns a list of document strings and document metadata (pd DataFrame). The Boolean argument determines whether or not the returned objects are exported to the current working directory"""

    file = open(f"{project_directory}cisi/CISI.ALL") #current_wording_dir/CISI.ALL
    contents = file.read()
    file.close()

    contents = contents.split('.I ')

    documents_df = {}
    documents = []

    for doc_and_metadata in contents[1:]:

        doc_and_metadata = doc_and_metadata.split("\n")
        doc_and_metadata = [x.strip() for x in doc_and_metadata]

        title_idx = doc_and_metadata.index('.T')
        author_idx = doc_and_metadata.index('.A')
        doc_idx = doc_and_metadata.index('.W')
        related_docs_idx = doc_and_metadata.index('.X')

        doc_id = int(doc_and_metadata[0])
        xref_docs = doc_and_metadata[related_docs_idx+1:-1]

        doc_title = doc_and_metadata[title_idx+1]
        doc_author = doc_and_metadata[author_idx+1]
        doc_text = ' '.join(doc_and_metadata[doc_idx+1:related_docs_idx])
        documents.append(doc_text) #doc index will be doc_id-1

        xref_doc_ids = []
        for doc in xref_docs:
            doc = doc.split('\t')
            xref_doc_ids.append(int(doc[0]))
        xref_doc_ids = set(xref_doc_ids)    

        doc_data = (doc_title, doc_author, xref_docs, doc_text)
        documents_df[doc_id] = doc_data

    documents_df = pd.DataFrame.from_dict(documents_df, orient='index', columns=['doc_title', 'doc_author', 'xref_docs', 'doc_text'])

    if save_documents:
        
        #writing documents out to a file (one per line)
        DATA_DIRECTORY = f"{project_directory}data"
        DOCS_FILE_PATH = DATA_DIRECTORY + '/document_text.txt'
        
        os.makedirs(DATA_DIRECTORY)
        outfile = open(DOCS_FILE_PATH, 'w')
        for line in documents:
            line = line + '\n'
            outfile.write(line)
        outfile.close()

        documents_df.to_csv(f"{project_directory}data/documents_data.txt", sep="|")

    return documents, documents_df

# EXAMPLE USAGE:
# documents, documents_df = parse_documents_file()

def parse_queries_file(project_directory: str, save_queries=False):
    """takes the source data’s directory as input, parses through the query data, and returns a list of query strings. The Boolean argument determines whether or not the returned list is exported to the current working directory"""
    
    clean_queries = []

    queryfile = open('/Users/marvazqu8/mv/winter23/csc575/final_project/cisi/CISI.QRY') #project/data/CISI.QRY
    queries = queryfile.read()
    queryfile.close()

    queries = queries.split('.I')
    queries = queries[1:] #first item is empty space
    
    for query in queries:
        query_and_metadata = query.split("\n")
        query_and_metadata = [q.strip() for q in query_and_metadata]

        #all have index and all have query
        query_id = int(query_and_metadata[0])
        query_idx = query_and_metadata.index('.W')

        try:
            #some have title
            query_title_idx = query_and_metadata.index('.T')
            #some have author 
            query_author_idx = query_and_metadata.index('.A')
            #some have .B?
            b_index = query_and_metadata.index('.B')

        except:
            #query will be everything from query_idx+1 to the end of list
            query_text = ' '.join(query_and_metadata[query_idx+1:])

            query_title_idx, query_author_idx, b_index = None, None, None

        if query_title_idx:
            query_text = ' '.join(query_and_metadata[query_idx+1:b_index])

        clean_queries.append(query_text)

    #remove trailing and leading spaces
    clean_queries = [x.strip() for x in clean_queries]

    #remove double spaces
    clean_queries = [x.replace("  ", " ") for x in clean_queries]

    if save_queries:
        # write out to file
        outfile = open(f"{project_directory}data/queries.txt", 'w+')
        for line in clean_queries:
            line = line + '\n'
            outfile.write(line)
        outfile.close()

    return clean_queries