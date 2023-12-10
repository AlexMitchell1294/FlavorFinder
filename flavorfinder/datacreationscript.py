import sqlite3
from search.search_engine import *
import pandas as pd
from itertools import combinations


def generate_substrings(text):
    words = text.split()
    substrings = [' '.join(words[i:j]) for i, j in combinations(range(len(words) + 1), 2)]
    return substrings


print("creating frame")
searchengine = SearchEngine()
# run this line if the pkl files have not beean built
print("creating inverse index")
# searchengine.create_inverse_index()
# searchengine.load_inverse_index()
# df = pd.read_csv("data/RAW_interactions.csv")
# print(len(df["user_id"].unique()))

# run these if sdb.sqlite3 has not been created
print("connecting to database")
conn = sqlite3.connect("db.sqlite3")
cur = conn.cursor()

# print("Creating recipes table")
# searchengine.recipes.to_sql("Recipes", conn)
# query = "DELETE FROM auth_user"
#
# cur.execute(query)
# conn.commit()

# print("creating users table")
# df = pd.read_csv("data/RAW_interactions.csv")
# col = df["user_id"].unique()
# for c in col:
#     cur.execute("INSERT INTO auth_user (username, password, is_superuser, is_active, is_staff, last_name, email, date_joined, first_name) VALUES (?, ?, 0 ,1 ,0, \"\", \"\", ?, \"\")", (str(c), str(c), "2023-12-05 18:30:04.597817"))
# conn.commit()
# col.to_sql("users", conn)
# for username in col:
#     if not User.objects.filter(username=username).exists():
#         User.objects.create_user(username=username, email=username, password=username)

# print("Creating lookup index")
# # cur.execute("CREATE TABLE search_index(name TEXT, recipe INTEGER, fullname TEXT, type TEXT)")
# for key in searchengine.inverse_ingredient_index.keys():
#     for recipe in searchengine.inverse_ingredient_index[key]:
#         result = generate_substrings(key)
#         # print(key, recipe)
#         for substring in result:
#             cur.execute("INSERT INTO search_index  VALUES(?, ?, ?, ?)", (substring, int(recipe), key, "ingredient"))
# conn.commit()
# for key in searchengine.inverse_tag_index.keys():
#     for recipe in searchengine.inverse_tag_index[key]:
#         result = generate_substrings(key)
#         # print(key, recipe)
#         for substring in result:
#             cur.execute("INSERT INTO search_index  VALUES(?, ?, ?, ?)", (substring, int(recipe), key, "tag"))
# conn.commit()
# for key in searchengine.inverse_minute_index.keys():
#     for recipe in searchengine.inverse_minute_index[key]:
#         # print(key, recipe)
#         cur.execute("INSERT INTO search_index  VALUES(?, ?, ?, ?)", (str(key), int(recipe), str(key), "minutes"))
# conn.commit()
cur.execute("SELECT * FROM RECIPES")
recipes = cur.fetchall()
for recipe in recipes:
    print(recipe[1])
    result = []
    if recipe[1]:
        result = [format_query(q) for q in recipe[1].split()]
    for substring in result:
        cur.execute("INSERT INTO search_index  VALUES(?, ?, ?, ?)", (substring, int(recipe[2]), recipe[1], "title"))

    # for recipe in searchengine.inverse_minute_index[key]:
    #     # print(key, recipe)
    #     cur.execute("INSERT INTO search_index  VALUES(?, ?, ?, ?)", (str(key), int(recipe), str(key), "minutes"))
conn.commit()
conn.close()
