#!/usr/bin/python

#imports 
import os, math, re, file_parsing, app_helpers
import numpy as np
import pandas as pd
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
np.seterr(divide='ignore', invalid='ignore')

def file_exists(path):
    '''checks that a file exists'''
    return os.path.isfile(path)

#main
if __name__== "__main__":
    
    #check if the inverted index file exists in the project folder (current directory)
    project_dir = str(os.getcwd()) + "/"

    if not file_exists(f"{project_dir}data/inv_idx.csv"): # parse all files & save them to be loaded in subsequent runs of the system 
        
        print("COULD NOT FIND INVERTED INDEX FILE IN THE PROJECT DATA DIRECTORY!")
        #read in documents, save them to a file 
        # (assuming that if the inverted index file isnt present then its the first time running the system)
        print("PARSING DOCUMENTS & STORING THEM IN THE PROJECT DATA DIRECTORY!")
        documents, documents_df = file_parsing.parse_documents_file(project_dir, save_documents=True)
        
        print("BUILDING INVERTED INDEX & SAVING IT TO THE PROJECT DATA DIRECTORY!")
        # TODO: consider renaming this function somehting like: "create_inv_index"
        token_dictionary, inverted_index = app_helpers.inv_indx_display(documents)

        #write the inverted index to a file
        inverted_index.to_csv(f"{project_dir}data/inv_idx.csv", sep='|')

        print("CREATING THE TERM-DOCUMENT MATRIX & SAVING IT TO THE PROJECT DATA DIRECTORY!")
        #create the doc-term matrix from the inverted index
        td_matrix = pd.DataFrame(0, index=inverted_index.index, columns=range(1, len(documents)+1))
        for term in token_dictionary.keys():
            for doc in token_dictionary[term].keys():
                if doc != 'Doc Freq' and doc != 'Total Freq':
                    td_matrix.loc[term, doc] = token_dictionary[term][doc]

        #remove rows with index = NaN from the td_matrix
        td_matrix = td_matrix.drop(td_matrix[td_matrix.index.isnull()].index)

        #write the td matrix to a file
        td_matrix.to_csv(f"{project_dir}data/td_matrix.csv")

    else:
        print("LOADING DOCUMENTS...")
        #load documents
        doc_file = open(f'{project_dir}data/document_text.txt')
        documents = doc_file.readlines()
        documents_df = pd.read_csv(f'{project_dir}data/documents_data.txt', sep="|", header=0, index_col=0)
        
        print("LOADING TERM-DOCUMENT MATRIX...")
        #load td matrix
        td_matrix = pd.read_csv(f'{project_dir}data/td_matrix.csv', header=0, index_col=0)
        #remove rows with index = NaN from the td_matrix
        td_matrix = td_matrix.drop(td_matrix[td_matrix.index.isnull()].index)
        
        print("LOADING INVERTED INDEX...")
        #load the inverted index
        inverted_index = pd.read_csv(f'{project_dir}data/inv_idx.csv', sep="|", header=0, index_col=0)

    print("APPLYING TF-IDF TRANSFORMATION TO THE TD-MATRIX...")
    if not file_exists(f"{project_dir}data/tfidf_td_matrix.csv"):
        tfidf_td_matrix, term_idfs = app_helpers.tf_idf(td_matrix, project_dir, save_dfs=True)
    
    else:
        tfidf_td_matrix = pd.read_csv(f"{project_dir}data/tfidf_td_matrix.csv", header=0, index_col=0, skiprows=[1])
        # tfidf_td_matrix = td_matrix.drop(tfidf_td_matrix[tfidf_td_matrix.index.isnull()].index)
        term_idfs = pd.read_csv(f"{project_dir}data/term_idf_values.csv", header=0, index_col=0)
        term_idfs = term_idfs.drop(term_idfs[term_idfs.index.isnull()].index)

    continue_running = True
    original_query = None
    rocchio_query = None
    rocchio_expansion = False
    sim_measure_map = {1: app_helpers.cosine_sim,
                       2: app_helpers.dot_product,
                       3: app_helpers.dice_coefficient,
                       4: app_helpers.jaccard_coefficient}

    while continue_running == True:
        ask_user_to_expand = True

        if rocchio_expansion == True:
            print(f'your original query: "{original_query}" has been updated using the relevance feedback you provided!')
            print(f'The Rocchio Method of query expansion reweights terms based on your feedback. This query is then used to find similar documents')
            query_vector = rocchio_query
            sim_measure = app_helpers.get_numeric_input("simmeasure")
            sim_measure = sim_measure_map[sim_measure]
            similar_documents = app_helpers.find_similar_docs(tfidf_td_matrix, query_vector, num_docs_to_return, sim_measure)

        else:
            #data is loaded at this point. ask user for a query?
            if original_query:
                print(f"ORIGINAL QUERY: {original_query}")
                user_query = app_helpers.get_query_input()
            else:
                user_query = app_helpers.get_query_input()
                original_query = user_query
            
            # sim_measure = app_helpers.get_similarity_measure_input()
            sim_measure = app_helpers.get_numeric_input("simmeasure")
            sim_measure = sim_measure_map[sim_measure]
            # num_docs_to_return = app_helpers.get_num_docs_input()
            num_docs_to_return = app_helpers.get_numeric_input("numdocs")
            
            #map query to a vector in the same space as the td matrix (query_vector) NOTE: QUERY IS STEMMED!
            #remove stopwords from the query (filtered_query)
            query_vector, filtered_query = app_helpers.string_process(user_query, tfidf_td_matrix.index)
            
            #tfidf transform query_vector
            tfidf_query = app_helpers.tfidf_transform_query(query_vector, term_idfs)

            #find simlar documents to the query
            similar_documents = app_helpers.find_similar_docs(tfidf_td_matrix, tfidf_query, num_docs_to_return, sim_measure)

        #display documents to the user
        print('-'*60)
        print(f"THESE ARE THE {num_docs_to_return} MOST RELEVANT DOCUMENTS TO YOUR QUERY:")
        # print(simlar_documents) #their IDs
        for i in range(1, num_docs_to_return+1):
            document_id = similar_documents[i-1]
            document_title = documents_df.loc[document_id, 'doc_title']
            print(f'#{i}. {document_title} (Doc ID: {document_id})')

        app_helpers.show_results(similar_documents, documents_df)

        if rocchio_expansion == True:
            new_search = app_helpers.get_yn_input("newsearch")
            if new_search == "N":
                continue_running == False
                break

            else:
                original_query = None
                rocchio_query = None
                rocchio_expansion = False
                ask_user_to_expand = False

        if ask_user_to_expand == True:
            #prompt the user asking if they want to expand their query
            expand_query = app_helpers.get_yn_input("expansion")

            if expand_query == 'Y':
                # expansion_method = app_helpers.get_expansion_method_input()
                expansion_method = app_helpers.get_numeric_input("expansion")

                if expansion_method == 1:
                    app_helpers.expand_query(filtered_query)
                else:
                    rocchio_query = app_helpers.rocchio_expand(tfidf_query, similar_documents, documents_df, tfidf_td_matrix)
                    rocchio_expansion = True

            #ask if they want to modify their query
            if rocchio_expansion == False:
                user_wants_to_continue = app_helpers.get_yn_input("continue")
                if user_wants_to_continue == 'N':
                    continue_running = False