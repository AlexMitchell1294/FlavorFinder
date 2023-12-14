# -*- coding: utf-8 -*-


import math
import random
import time

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm.notebook as tqdm  # progress bars
# import implicit # Fast, sparse ALS implementation
from search.search_engine import format_query

from itertools import product

from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
# recipes = pd.read_csv('data/PP_recipes.csv')
# ingr_map = pd.read_pickle('data/ingr_map.pkl')
raw_recipes = pd.read_csv('data/RAW_recipes.csv')

raw_recipes.head()

# ingr_map.head()

# recipes.head()


def get_tf_idf_embeddings(recipe_ingr_xref):
    '''

    '''

    # Create the document frequency matrix
    ingr_freq = csr_matrix((np.ones(len(recipe_ingr_xref)), (recipe_ingr_xref['i'], recipe_ingr_xref['ingr'])))

    # Generate the tf-idf embeddings from the document frequency matrix
    tfidf = TfidfTransformer()
    tf_idf_embeddings = tfidf.fit_transform(ingr_freq)

    return tf_idf_embeddings


def cosine_similar_predictions(embeddings, query):
    '''
    embeddings is a matrix of shape (num_recipes, num_features)
    query is a list of length num_features

    Returns an array of cosine similarity scores between the query and each row of embeddings
    '''

    # Ensure that the query is a 2D array (reshape if necessary)
    query = query.reshape(1, -1)

    # Compute cosine similarity between the query and each row of the embeddings matrix
    similarity_scores = cosine_similarity(embeddings, query).squeeze()

    return similarity_scores


def make_recipe_ingr_xref(recipes):
    '''
    recipes is the pandas dataframe for preprceossed recipes. The ingredient_ids column should store lists of integers

    Returns new pandas dataframe, a cross-reference table for all recipes and ingredients
    '''

    recipe_ids = []
    ingr_ids = []
    index = 0
    for row in tqdm.tqdm(recipes['ingredients'].index):
        for ingr_id in recipes['ingredients'][row]:
            ingr_id = format_query(ingr_id)
            recipe_ids.append(recipes.loc[row, 'id'])
            if ingr_id not in ingredient_ids.keys():
                ingredient_ids[ingr_id] = index
                index += 1
            ingr_ids.append(ingredient_ids[ingr_id])
        # for ingr_id in recipes['tags'][row]:
        #     recipe_ids.append(recipes.loc[row, 'id'])
        #     if ingr_id not in ingredient_ids.keys():
        #         ingredient_ids[ingr_id] = index
        #         index += 1
        #     ingr_ids.append(ingredient_ids[ingr_id])


    return pd.DataFrame.from_dict({'i': recipe_ids, 'ingr': ingr_ids})

ingredient_ids = {}
raw_recipes['ingredients'] = raw_recipes['ingredients'].map(
    lambda str: [ingr_id for ingr_id in str[1:-1].split(', ')])
# raw_recipes['tags'] = raw_recipes['tags'].map(
#     lambda str: [tag for tag in str[1:-1].split(', ')])

# Create a cross-reference tables for all recipes and ingredients
recipe_ingr_xref = make_recipe_ingr_xref(raw_recipes)

# Get the tf-idf embeddings
# tf_idf_embeddings = joblib.load('tfidf_transformer.pkl')
tf_idf_embeddings = get_tf_idf_embeddings(recipe_ingr_xref)

# Merge preprocessed and raw recipe tables
full_recipes = raw_recipes#recipes.merge(raw_recipes, left_on='id', right_on='id')


def similar_recipes(embeddings, recipe_id, k):
    # Values for each recipe based on similarity to given recipe
    similarities = cosine_similar_predictions(embeddings, embeddings[recipe_id])

    # Get reccomendations, filtering out the queries recipe
    return get_reccomendations(similarities, k, filtered_recipe_ids=[recipe_id])


def get_reccomendations(predictions, k, filtered_recipe_ids=[], filter_val=-100):
    # Replace all recipes to be filtered out with the filter value, which should be a lower value
    predictions[filtered_recipe_ids] = filter_val

    # Get the top k reccomendations for the user (NOTE: these are not in order)
    unsorted_recs = np.argpartition(predictions, -k)[-k:]

    # Sort the reccomendations by their predictions values, in descending order
    sorted_recs = sorted(unsorted_recs, key=lambda rec: predictions[rec], reverse=True)

    return sorted_recs


# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# simple_ingr_map = ingr_map[['id', 'replaced']].set_index('id').drop_duplicates().to_dict()['replaced']
# recipes['ingredients'] = recipes['ingredient_ids'].map(
#     lambda ingredient_ids: [simple_ingr_map[ingregient_id] for ingregient_id in ingredient_ids])
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(recipes['ingredients'])]

# model = Doc2Vec(vector_size=5, window=2, min_count=1, workers=4)
# model.build_vocab(corpus_iterable=documents, progress_per=1000)
#
# model.train(corpus_iterable=documents, total_examples=model.corpus_count, epochs=30)

joblib.dump(tf_idf_embeddings, "tfidf_transformer.pkl")

recipe_id = 124722
k = 10

start = time.time()
# Content-based recommendations
content_recs = similar_recipes(tf_idf_embeddings, recipe_id, k)

print(f'If you liked {full_recipes[full_recipes["id"] == recipe_id].iloc[0]["name"]}, you might also like:')
for i, rec in enumerate(content_recs):
    print(full_recipes[full_recipes['id'] == rec].iloc[0]['name'])
print(time.time() - start)

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score


# Function to evaluate Precision at K (P@K)
def precision_at_k(actual, predicted, k):
    actual_relevant_recipes = set(actual)
    content_recs = predicted[:k]
    num_relevant_at_k = len(set(content_recs).intersection(actual_relevant_recipes))

    return num_relevant_at_k / k  # Precision at K


# Function to evaluate Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(actual, predicted):
    for i, rec in enumerate(predicted):
        if rec in actual:
            return 1 / (i + 1)
    return 0


# Initialize lists to store evaluation results for all recipes
all_p_at_k = []
all_mrr = []

# # Iterate over each recipe for evaluation
# for test_recipe_id in tqdm.tqdm(recipes['i']):
#     # Replace this with your actual ground truth data
#     actual_relevant_recipes = [1, 5, 8, ...]  # Replace with the actual relevant recipes for your evaluation
#
#     # Get content-based recommendations
#     content_recs = similar_recipes(tf_idf_embeddings, test_recipe_id, k=k)
#
#     # Ensure the correct length for actual_relevant_recipes
#     actual_relevant_recipes = actual_relevant_recipes[:k]
#
#     # Ensure the correct length for content_recs
#     content_recs = content_recs[:k]
#
#     # Evaluate Precision at K and MRR
#     p_at_k = precision_at_k(actual_relevant_recipes, content_recs, k=k)
#     mrr = mean_reciprocal_rank(actual_relevant_recipes, content_recs)
#
#     # Append results to lists
#     all_p_at_k.append(p_at_k)
#     all_mrr.append(mrr)
#
# # Calculate the mean of all evaluation metrics
# mean_p_at_k = np.mean(all_p_at_k)
# mean_mrr = np.mean(all_mrr)
#
# print(f'Mean Precision at K: {mean_p_at_k}')
# print(f'Mean Mean Reciprocal Rank: {mean_mrr}')
