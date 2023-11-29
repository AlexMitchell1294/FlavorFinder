import sqlite3
from search.search_engine import SearchEngine

searchengine = SearchEngine()
#run this line if the pkl files have not beean built
# searchengine.create_inverse_index()

#run these if sdb.sqlite3 has not been created
# conn = sqlite3.connect("db.sqlite3")
# searchengine.recipes.to_sql("Recipes", conn)
# conn.close()
