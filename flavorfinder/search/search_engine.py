import sqlite3

import pandas as pd
import pickle as pk
import ast
import re
import nltk
from django.db.models import When, Case, IntegerField
from nltk.stem import PorterStemmer
from collections import Counter
from itertools import combinations
import time
import tensorflow_recommenders as tfrs
import tensorflow as tf
from tqdm import tqdm


interaction_data = pd.read_csv("data/RAW_interactions.csv")
recipe_data = pd.read_csv("data/RAW_recipes.csv")

interaction_train = pd.read_csv("data/interactions_train.csv")
interaction_test = pd.read_csv("data/interactions_test.csv")
interaction_data = interaction_data.astype({'user_id': 'string', 'recipe_id': 'string'})
interaction_train = interaction_train.astype({'user_id': 'string', 'recipe_id': 'string'})
interaction_test = interaction_test.astype({'user_id': 'string', 'recipe_id': 'string'})

uniqueUserIds = interaction_data.user_id.unique()
uniqueFoodIds = interaction_data.recipe_id.unique()


def text_search(query, user_id):
    time1 = time.time()
    all_q = generate_substrings(query)
    tokens = [format_query(q) for q in all_q]
    conn = sqlite3.Connection("db.sqlite3")
    cur = conn.cursor()
    sql = 'SELECT name, recipe FROM search_index WHERE name IN ({0})'.format(', '.join('?' for _ in tokens))
    cur.execute(sql, tokens)

    rows = cur.fetchall()
    conn.close()
    search_results = {}
    for row in rows:
        if row[0] in search_results.keys():
            search_results[row[0]].append(row[1])
        else:
            search_results[row[0]] = [row[1]]
    # print(search_results)
    # print(time.time() - time1)
    # time2 = time.time()
    # time2 = time.time()
    # if token in search_results.keys():
    #     search_results[token].extend([item[0] for item in row])
    # else:
    #     search_results[token] = [item[0] for item in row]
    # print(search_results)

    ret_results = set(list(search_results.values())[0])
    for i in range(1, len(search_results.keys())):
        if list(search_results.values())[i]:
            set2 = set(list(search_results.values())[i])
            ret_results = ret_results.intersection(set2)
    # conn.close()

    ############
    # plug into model

    ############
    time4 = time.time()
    print(time4 - time1)
    return list(ret_results)


def format_query(q):
    stemmer = PorterStemmer()
    q = re.sub('[^a-z0-9 *]', ' ', q.lower())
    q = stemmer.stem(q)
    return q


def generate_substrings(text):
    words = text.split()
    substrings = [' '.join(words[i:j]) for i, j in combinations(range(len(words) + 1), 2)]
    return substrings


def predict_all_user_food_pairs(model, uniqueUserIds, uniqueFoodIds, batch_size=64):
    predictions_df = pd.DataFrame(columns=["user_id", "recipe_id", "prediction"])

    i = 0
    for user_batch in tf.data.Dataset.from_tensor_slices(uniqueUserIds).batch(batch_size):
        user_batch = tf.cast(user_batch, tf.string)
        user_embeddings = model.ranking_model.user_embeddings(user_batch)

        for food_batch in tf.data.Dataset.from_tensor_slices(uniqueFoodIds).batch(batch_size):
            food_batch = tf.cast(food_batch, tf.string)
            food_embeddings = model.ranking_model.product_embeddings(food_batch)

            user_embeddings_expanded = tf.repeat(user_embeddings, len(food_batch), axis=0)
            food_embeddings_expanded = tf.tile(food_embeddings, [len(user_batch), 1])

            batch_predictions = model.ranking_model.ratings(
                tf.concat([user_embeddings_expanded, food_embeddings_expanded], axis=1)
            )

            batch_df = pd.DataFrame({
                "user_id": tf.repeat(user_batch, len(food_batch)).numpy(),
                "recipe_id": tf.tile(food_batch, [len(user_batch)]).numpy(),
                "prediction": batch_predictions.numpy().flatten()
            })

            predictions_df = pd.concat([predictions_df, batch_df], ignore_index=True)

    return predictions_df


def sort_recipes(recipe_ids, userid):
    from .models import Recipes
    loaded = tf.saved_model.load("models/")
    recipe_ids = [str(ele) for ele in recipe_ids]
    ranked_ids = predict_all_user_food_pairs(loaded, [userid], recipe_ids)
    ranked_ids = ranked_ids.sort_values(by="prediction")
    ranked_ids = ranked_ids["recipe_id"].tolist()[:1000]

    whens = [When(id=id_val, then=pos) for pos, id_val in enumerate(ranked_ids)]

    # Create a Case expression using the When expressions
    case_expression = Case(*whens, default=0, output_field=IntegerField())
    return Recipes.objects.filter(id__in=ranked_ids).order_by(case_expression)


class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=uniqueUserIds, mask_token=None),
            # add addional embedding to account for unknow tokens
            tf.keras.layers.Embedding(len(uniqueUserIds) + 1, embedding_dimension)
        ])

        self.product_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=uniqueFoodIds, mask_token=None),
            # add addional embedding to account for unknow tokens
            tf.keras.layers.Embedding(len(uniqueFoodIds) + 1, embedding_dimension)
        ])
        # Set up a retrieval task and evaluation metrics over the
        # entire dataset of candidates.
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, userId, foodId):
        user_embeddings = self.user_embeddings(userId)
        food_embeddings = self.product_embeddings(foodId)
        return self.ratings(tf.concat([user_embeddings, food_embeddings], axis=1))


# Build a model.
class FoodModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model(features["userID"], features["foodID"])

        return self.task(labels=features["rating"], predictions=rating_predictions)


class SearchEngine:
    def __init__(self):
        self.inverse_ingredient_index = {}
        self.inverse_tag_index = {}
        self.inverse_minute_index = {}
        self.stemmer = PorterStemmer()
        self.recipes = pd.read_csv("data/RAW_recipes.csv")

    def search(self, list_of_ingredients, amount_of_ingredient_match, list_of_tags, amount_of_tags_match, min_time,
               max_time):
        start_time = time.time()
        search_result_ingredient = self.search_by_ingredients(list_of_ingredients, amount_of_ingredient_match)
        search_result_tag = self.search_by_tags(list_of_tags, amount_of_tags_match)
        set1 = set(search_result_ingredient)
        set2 = set(search_result_tag)
        search_results_filtered_time = self.get_recipes_by_minutes(min_time, max_time, list(set1.intersection(set2)))
        return search_results_filtered_time, time.time() - start_time

    def pretty_return(self, list_of_recipe_id):
        # for recipe in list_of_recipe_id:
        pass

    def create_inverse_index(self):
        self.recipes["ingredients"] = self.recipes["ingredients"].apply(ast.literal_eval)
        self.recipes["tags"] = self.recipes["tags"].apply(ast.literal_eval)
        for i, list_of_recipes in enumerate(self.recipes.iterrows()):

            for ingredient in self.recipes.iloc[i]["ingredients"]:
                ingredient = format_query(ingredient)
                if ingredient in self.inverse_ingredient_index.keys():
                    self.inverse_ingredient_index[ingredient].append(self.recipes.iloc[i]["id"])
                else:
                    self.inverse_ingredient_index[ingredient] = [self.recipes.iloc[i]["id"]]
            for tag in self.recipes.iloc[i]["tags"]:
                tag = format_query(tag)
                if tag in self.inverse_tag_index.keys():
                    self.inverse_tag_index[tag].append(self.recipes.iloc[i]["id"])
                else:
                    self.inverse_tag_index[tag] = [self.recipes.iloc[i]["id"]]

            if self.recipes.iloc[i]["minutes"] in self.inverse_minute_index:
                self.inverse_minute_index[self.recipes.iloc[i]["minutes"]].append(self.recipes.iloc[i]["id"])
            else:
                self.inverse_minute_index[self.recipes.iloc[i]["minutes"]] = [self.recipes.iloc[i]["id"]]

        with open('data/inverse_ingredient_index.pkl', 'wb') as f:
            pk.dump(self.inverse_ingredient_index, f)
        with open('data/inverse_tag_index.pkl', 'wb') as f:
            pk.dump(self.inverse_tag_index, f)
        with open('data/inverse_minute_index.pkl', 'wb') as f:
            pk.dump(self.inverse_minute_index, f)

    def load_inverse_index(self):
        with open('data/inverse_ingredient_index.pkl', 'rb') as f:
            self.inverse_ingredient_index = pk.load(f)
        with open('data/inverse_tag_index.pkl', 'rb') as f:
            self.inverse_tag_index = pk.load(f)
        with open('data/inverse_minute_index.pkl', 'rb') as f:
            self.inverse_minute_index = pk.load(f)

    def get_ingredient_list(self, ingredient):
        ingredient = format_query(ingredient)
        return self.inverse_ingredient_index[ingredient]

    def get_recipe(self, recipe_id):
        return self.recipes.loc[self.recipes["id"] == recipe_id]

    def search_by_ingredients(self, list_of_ingredients, accuracy):
        """
        get a list of recipe_id's that fit the search query

        Parameters:
        list_of_ingredients (list of strings): A list of ingredients a user wants to use
        accuracy (int): how many items it should match, if len(list_of_ingredients)==3 and accuracy is 3 it would mean
            it must have all items in list_of_ingredients
            where 2 means it only needs to match 2 of the ingredients (this could be any mix of 2 items in
            list_of_ingredients

        Returns:
            list of recipes_ids

        """
        tokenized = [format_query(ingredient) for ingredient in list_of_ingredients]

        recipe_counts = Counter()
        for token in tokenized:
            recipe_counts.update(self.inverse_ingredient_index.get(token, []))

        return [key for key, value in recipe_counts.items() if value >= accuracy]

    def search_by_tags(self, list_of_tags, accuracy):
        tokenized = [format_query(tag) for tag in list_of_tags]

        recipe_counts = Counter()
        for token in tokenized:
            recipe_counts.update(self.inverse_tag_index.get(token, []))

        return [key for key, value in recipe_counts.items() if value >= accuracy]

    def get_recipes_by_minutes(self, min_time, max_time, recipe_list):
        timings = set()
        if max_time >= 130:
            max_time = max(self.inverse_minute_index.values())
        for t in self.inverse_minute_index.keys():
            if min_time <= t <= max_time:
                timings.update(self.inverse_minute_index[t])

        return list(set(recipe_list) & timings)
