#!/usr/bin/python

#imports 
import os, math, re, file_parsing, string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords , wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

def tf_idf(td_mat: pd.DataFrame, project_directory:str, save_dfs=False):
    """converts raw term counts of a term-document matrix to non-normalized tf-idf weights"""
    
    term_idf = {}
    
    for term in td_mat.index:
        row = td_mat.loc[term,:]
        doc_freq = sum([1 for freq in row if freq > 0])
        idf = len(row) / doc_freq
        idf = math.log2(idf)
        
        term_idf[term] = idf
        td_mat.loc[term,:] = row * idf
    
    term_idf_df = pd.DataFrame.from_dict(term_idf, orient='index', columns=['idf_value'])

    if save_dfs:
        td_mat.to_csv(f"{project_directory}data/tfidf_td_matrix.csv")
        term_idf_df.to_csv(f"{project_directory}data/term_idf_values.csv")

    return td_mat, term_idf_df

def string_process(strg: str, td_matrix_index): 
    """takes a query and returns a vector of term frequencies for 
       that query in the term_doc_matrix feature space"""
    
    punctuations = list(string.punctuation)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(strg) #tokenization
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words] #remove stopwords
    stemmed_sentence = [ps.stem(s) for s in filtered_sentence] #stemmed
    no_punct = [i.strip(''.join(punctuations)) for i in stemmed_sentence if i not in punctuations]
    
    query_vec = pd.DataFrame(0,
                             index = ['query'],
                            #  columns = sorted(set(td_matrix_index.tolist() + no_punct)))
                            columns = td_matrix_index)
    try:
        for item in no_punct:
            query_vec.at['query', item] += 1
    except KeyError:
        pass
        
    terms_to_drop = [x for x in query_vec.columns if x not in td_matrix_index]
    query_vec = query_vec.drop(terms_to_drop, axis=1)
    
    return query_vec, filtered_sentence

def tfidf_transform_query(query: pd.DataFrame, term_idfs: pd.DataFrame):
    """takes a vectorized query and term IDFs (both pd DataFrames) as input, applies the weights to the query, and returns a normalized version of the query vector (pd DataFrame)"""
    
    for term in term_idfs.index:
        query.loc["query", term] *= term_idfs.loc[term, "idf_value"]
        
    return query

def cosine_sim(doc1: pd.DataFrame, doc2: pd.DataFrame):
    """compute the cosine similarity of 2 documents"""
    
    doc1, doc2 = np.array(doc1), np.array(doc2)
    
    #find the vector norm for each instance in training_data
    doc1_norm = np.linalg.norm(doc1)
    doc2_norm = np.linalg.norm(doc2)

    #compute cosine: divide the dot product of instance & every instance in training data, then divide by product of the 2 vector norms
    numerator = np.dot(doc1, doc2) 
    denominator = (doc1_norm * doc2_norm)
    cosine_similarity = numerator / denominator

    return cosine_similarity

def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)

def dice_coefficient(vector1, vector2):
    """computes dice’s similarity between two vectors"""

    numerator = 2 * dot_product(vector1, vector2)
    denominator = sum(vector1.loc['query', :] ** 2) + sum(vector2 ** 2)
    sim = numerator / denominator
    
    return sim

def jaccard_coefficient(vector1, vector2):
    """computes jaccard’s similarity between two vectors"""
    
    numerator = dot_product(vector1, vector2)
    denominator = sum(vector1.loc['query', :] ** 2) + sum(vector2 ** 2)
    denominator -= numerator
    
    sim = numerator / denominator
    
    return sim

def find_similar_docs(tfidf_term_doc_matrix:pd.DataFrame, query: pd.DataFrame, n: int, sim_measure):
    """takes a normalized term-document matrix (pd DataFrame), normalized query vector (pd DataFrame), number of similar documents, N, and a similarity measure as input and returns N most similar documents to the inputted query (list of their IDs)"""

    similarities = np.empty(0)

    for document_id in tfidf_term_doc_matrix.columns:
        document = tfidf_term_doc_matrix.loc[:, document_id]
        similarity = sim_measure(query, document)
        
        similarities = np.append(similarities, similarity)
        
    n_most_similar = similarities.argsort()[::-1][:n]

    doc_ids = tfidf_term_doc_matrix.columns[n_most_similar]
    doc_ids = [int(id) for id in doc_ids]

    return doc_ids

def expand_query(query: list):
    """takes a query (list of strings with stop words filtered out) as input and displays synonyms of each word"""
    
    for word in query:
        current_synonyms = []
        print(f"Consider replacing '{word}' in the original query with one of the following terms:")
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                current_synonyms.append(lemma.name())
        print(', '.join(set([x for x in current_synonyms if x!= word])))
        print('-'*25)

def centroid(docs: list, dt_mat:pd.DataFrame):
    """takes a list of document IDs and a document-term matrix (pd DataFrame) as input and returns the average of the corresponding document vectors (pd Series)"""
    
    docs = [str(doc) for doc in docs]
    documents = dt_mat.loc[docs, :]
    centroid = documents.sum(axis=0) / len(docs)
    return(centroid)
    
def modify_query(Q: pd.DataFrame, R: list, NR: list, alpha: float, beta:float, dt_matrix: pd.DataFrame):
    """takes the vectorized query (pd DataFrame), list of document IDs of relevant and irrelevant documents, alpha and beta values, and a document-term matrix as input and returns a modified query (np array) that incorporates relevance feedback via the Rocchio method"""

    relevant_centroid, nonrelevant_centroid = centroid(R, dt_matrix), centroid(NR, dt_matrix)
    updated_query = Q + (alpha * relevant_centroid) - (beta * nonrelevant_centroid)
    return updated_query

def rocchio_expand(query_vec, recommended_documents, documents_and_metadata, td_matrix:pd.DataFrame):
    """takes a vectorized query (pd DataFrame), list of recommended documents, document metadata (pd DataFrame), and a term-document matrix as input and returns an updated query (np array) that is constructed via the modify_query function"""

    relevant_documents = []
    nonrelevant_docs = []

    for document_id in recommended_documents:
        document_text = documents_and_metadata.loc[document_id, 'doc_text']
        document_title = documents_and_metadata.loc[document_id, 'doc_title']
        print(f"Document ID: {document_id}")
        print(f"Document Title: {document_title}")
        print(document_text)

        doc_is_relevant = input("Is this document relevant to your search? (Y/N)")
        if doc_is_relevant.upper() == 'Y':
            relevant_documents.append(document_id)
        else:
            nonrelevant_docs.append(document_id)

    updated_query = modify_query(query_vec, relevant_documents, nonrelevant_docs, 0.5, 0.25, td_matrix.T)
    return updated_query

def inv_indx_display(documents: list):
    """takes a list of documents as pre-processed string items as input\
    and returns an inverted index (as a pandas DataFrame) as output"""
    
    #token dictionary
    token_dict = {}
    
    stop_words = set(stopwords.words('english'))
    punctuations = list(string.punctuation)
    punctuations.append('')

    for i in range(len(documents)):
        word_tokens = word_tokenize(documents[i]) #tokenization
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words] #remove stopwords
        stemmed_sentence = [ps.stem(s) for s in filtered_sentence] #stemmed
        no_punct = [i.strip(''.join(punctuations)) for i in stemmed_sentence if i not in punctuations] #remove punctuation
        #no_num = [k for k in no_punct if not k[0].isdigit()]
        for word in no_punct:
            #new word, new document (entirely new word)
            if word not in token_dict:
                token_dict[word] = {}
                token_dict[word]['Doc Freq'] = 1
                token_dict[word]['Total Freq'] = 1
                token_dict[word][i+1] = 1
            else:
                token_dict[word]['Total Freq'] += 1
                #old word, new document
                if i+1 not in token_dict[word]:
                    token_dict[word]['Doc Freq'] += 1
                    token_dict[word][i+1] = 1
                else:
                    #old word, old document
                    token_dict[word][i+1] += 1

    #token inverted index via pandas DataFrame
    filtered_tokens = sorted(token_dict.keys())
    token_index = pd.DataFrame(index = filtered_tokens,
                columns = ['Total Frequency', 'Document Frequency', 'Postings (Doc-ID, Count)'])

    token_index['Postings (Doc-ID, Count)'] = ''

    #fill in inverted index
    for item in filtered_tokens:
        token_index.loc[item]['Total Frequency'] = token_dict[item]['Total Freq']
        token_index.loc[item]['Document Frequency'] = token_dict[item]['Doc Freq']

    for item in filtered_tokens:
        for i in range(1, len(documents)+1):
            if i in token_dict[item]:
                token_index.loc[item]['Postings (Doc-ID, Count)'] += '({},{}) '\
                .format(i, token_dict[item][i])
                
    return token_dict, token_index

def get_query_input():
    """prompts the user for a valid query"""
    while True:
        query = input("Enter a search query: ")
        if len(query) < 2:
            print("!!!Please enter a valid query!!!")

        else:
            break
    return query

def get_yn_input(input_type: str):
    """prompts the user for a “Y” (yes) or “N” (no) response to specific questions"""

    input_strings = {'newsearch' : "Your query was expanded using the Rocchio Method.\nWould you like to start a new search? (!CURRENT SEARCH WILL END!) (Y/N): ",
              "expansion" : "Would you like to expand your query? (Y/N): ",
              "continue"  : "Would you like to continue? (Y/N): ",
              "results"   : "Would you like to see document information for any of your results? (Y/N): "}
    
    while True:
        user_input = input(f"{input_strings[input_type]}")
        user_input = user_input.upper()
        if user_input not in ['Y', 'N']:
            print('Please enter a valid response')
        else:
            break
    return user_input


def get_numeric_input(input_type: str, document_ids=None):
    """ prompts the user for quantitative response to specific questions"""

    input_strings ={"numdocs" :"How many results would you like to see? (1-20): ",
                    "simmeasure": "Choose a similarity measure to use for your search:\n1.Cosine Similarity\n2.Dot Product\n3.Dice Coefficient\n4.Jaccard Coefficient (1-4): ",
                    "expansion" : "Which method would you like to use to expand your query?: \n1. Using synonyms (found with WordNet)\n2. Input Relevance feedback (Rocchio Method)\nQuery Expansion method choice (1 or 2): ",
                    "results"   : "Please choose a document to see more information about (Doc ID #): "}
    
    while True:
        try:
            user_input = int(input(f"{input_strings[input_type]}"))
        except ValueError:
            print("!!!Please enter a valid response!!!")
            continue

        if input_type == "numdocs":
            if user_input not in range(1,20):
                print("!!!Please specify a valid number of results!!!")
                continue
            else:
                break

        elif input_type == 'simmeasure':
            if user_input not in range(1,5):
                print("!!!Please enter a valid choice for simlarity measure!!!")
                continue
            else:
                break

        elif input_type == 'expansion':
            if user_input not in [1,2]:
                print("!!!Please choose a valid expansion method!!!")
                continue
            else:
                break

        elif document_ids and input_type =='results':
            if user_input not in document_ids:
                print("Please choose a valid document from your search")
                continue
            else:
                break

        # else:
        #     break

    return user_input

def show_results(similar_doc_ids: list, document_data: pd.DataFrame):
    """takes a list of similar document IDs and document metadata (pd DataFrame) as input and displays the document metadata for the corresponding documents"""

    while True:
        see_results = get_yn_input("results")
        if see_results == 'N':
            break
        else:
            doc_id_to_show = get_numeric_input("results", similar_doc_ids)
            document_title = document_data.loc[doc_id_to_show, 'doc_title']
            document_author = document_data.loc[doc_id_to_show, 'doc_author']
            document_text = document_data.loc[doc_id_to_show, 'doc_text']
            doc_ranking_in_search = similar_doc_ids.index(doc_id_to_show) + 1
            print(f'#{doc_ranking_in_search}. {document_title} (Doc. #{doc_id_to_show})')
            print(f'Author: {document_author}')
            print(f'Document Text: {document_text}')