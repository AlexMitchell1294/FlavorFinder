import pandas as pd
import pickle as pk
import ast
import re
import nltk
from nltk.stem import PorterStemmer
from collections import Counter
import time


class SearchEngine:
    def __init__(self):
        self.inverse_ingredient_index = {}
        self.inverse_tag_index = {}
        self.inverse_minute_index = {}
        self.stemmer = PorterStemmer()
        self.recipes = pd.read_csv("data/RAW_recipes.csv")

    def search(self, list_of_ingredients, amount_of_ingredient_match, list_of_tags, amount_of_tags_match, min_time, max_time):
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
                ingredient = self.format_query(ingredient)
                if ingredient in self.inverse_ingredient_index.keys():
                    self.inverse_ingredient_index[ingredient].append(self.recipes.iloc[i]["id"])
                else:
                    self.inverse_ingredient_index[ingredient] = [self.recipes.iloc[i]["id"]]
            for tag in self.recipes.iloc[i]["tags"]:
                tag = self.format_query(tag)
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
        ingredient = self.format_query(ingredient)
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
        tokenized = [self.format_query(ingredient) for ingredient in list_of_ingredients]

        recipe_counts = Counter()
        for token in tokenized:
            recipe_counts.update(self.inverse_ingredient_index.get(token, []))

        return [key for key, value in recipe_counts.items() if value >= accuracy]

    def search_by_tags(self, list_of_tags, accuracy):
        tokenized = [self.format_query(tag) for tag in list_of_tags]

        recipe_counts = Counter()
        for token in tokenized:
            recipe_counts.update(self.inverse_tag_index.get(token, []))

        return [key for key, value in recipe_counts.items() if value >= accuracy]
    def format_query(self, q):
        q = q.lower()
        q = re.sub('[^a-z0-9 *]', ' ', q.lower())
        q = self.stemmer.stem(q)
        return q

    def get_recipes_by_minutes(self, min_time, max_time, recipe_list):
        timings = set()
        for t in self.inverse_minute_index.keys():
            if min_time <= t <= max_time:
                timings.update(self.inverse_minute_index[t])

        return list(set(recipe_list) & timings)
