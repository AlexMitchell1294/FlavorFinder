from searchengine.search import SearchEngine
from sqlalchemy import create_engine
import sqlite3
conn = sqlite3.connect("db.sqlite3")
c = conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS recipes (name text)')
conn.commit()

searchEngine = SearchEngine()

# searchEngine.create_inverse_index()

# searchEngine.load_inverse_index()
# searchEngine.recipes.to_sql("db.sqlite3")
searchEngine.recipes["name"].to_sql(name='recipes', con=conn)

# print(searchEngine.inverse_index)


# result = searchEngine.search(["potatoes", "firm tofu", "butter"], len(["potatoes", "firm tofu", "butter"]),
#                     ['japanese', 'seafood'], len(['japanese', 'seafood']), 30, 60)
# print(searchEngine.get_recipe(result[0][0]), result[1])



# print(len(searchEngine.search_by_ingredients(["carrot", "chicken", "rice"], 3)))
# print(len(searchEngine.search_by_ingredients(["carrot", "chicken", "rice"], 2)))
# print(len(searchEngine.search_by_ingredients(["carrot", "chicken", "rice"], 1)))

# print(len(search_result))
# search_result_minute_filtered = searchEngine.get_recipes_by_minutes(30, 60, search_result)
# print(search_result_minute_filtered)
# print(len(search_result_minute_filtered))

# print(searchEngine.inverse_tag_index.keys())
