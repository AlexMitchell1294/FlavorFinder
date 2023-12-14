from collections import defaultdict
import pandas as pd
import csv
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
# chunksize = 5000  # Read 1000 rows per chunk

# Load the CSV file into chunks
# data = np.genfromtxt('RAW_interactions.csv',filling_values=0)

#data = pd.read_csv('RAW_interactions.csv')
# my_dataframe = pd.DataFrame(data)
# #we create a utility matrix and fill the missing values with 0
# user_id=np.array(data['user_id'],dtype='int64')
# recipe_id=np.array(data['recipe_id'],dtype='int64')
# matrix = my_dataframe.pivot_table(index='recipe_id', columns='user_id', values='rating').fillna(0)
# print(matrix)
# #
# chunks = []
# ratings=np.array(data['rating'])
# user_id=np.array(data['user_id'])
# recipe_id=np.array(data['recipe_id'])
# # Process each chunk
# # for chunk in data:
# #     # Perform data exploration or analysis on each chunk
# # ratings=np.append(ratings,data['rating'])
# # user_id=np.append(user_id,data['user_id'])
# # recipe_id=np.append(recipe_id,data['recipe_id'])
#     # chunks.append(chunk)
# print(user_id)
# print(recipe_id)
# print(ratings)
# # combined_df = pd.concat(chunks, ignore_index=True)
# # print(combined_df)
# print(len(recipe_id))
# print(len(user_id))
# sparse_matrix = sparse.csr_array((ratings,(recipe_id,user_id)))
# #print(sparse_matrix)
# # column_sums = sparse_matrix.sum(axis=0)
# # print(column_sums)
# s=sparse_matrix.get_shape()
# print(s)
# sparse_matrix=sparse_matrix.tocoo()
# print(sparse_matrix)
#
# # for i in range(0,len(recipe_id)):
# # similarities_sparse = cosine_similarity(sparse_matrix)
# # print(format(similarities_sparse))

f= open('test.csv', 'r' , encoding='utf-8-sig')
header = f.readline().strip().split(',')
#del header[-1]
dataset = []
# Print the header and fields
print("Header:", header)
for line in f:
    fields = line.strip().split(',')
    #del fields[-1]
    d = dict(zip(header, fields))
    dataset.append(d)

df = pd.DataFrame(dataset)
ratings=df['rating'].tolist()
ratings = [eval(i) for i in ratings]
user_id=df['user_id'].tolist()
user_id = [eval(i) for i in user_id]
recipe_id=df['recipe_id'].tolist()
recipe_id = [eval(i) for i in recipe_id]
# print(df)
print(dataset)
# print(len(dataset))
matrix = sparse.coo_matrix((ratings, (recipe_id, user_id)))
# #matrix = sparse.csr_array((ratings,(recipe_id,user_id)))
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for d in dataset:
    user= int(d['user_id'])
    recipe =int(d['recipe_id'])
    reviewsPerUser[user].append(d)
    reviewsPerItem[recipe].append(d)
print(reviewsPerItem)
print(reviewsPerUser)

ratingMean = sum([i for i in ratings]) / len(dataset)
#print(ratingMean)
labels = [int(d['rating']) for d in dataset]
def Cosine(s1, s2):
    dot_product = s1.dot(s2.T).toarray()[0, 0]
    norm_s1 = np.sqrt(s1.power(2).sum())
    norm_s2 = np.sqrt(s2.power(2).sum())

    if norm_s1 == 0 or norm_s2 == 0:
        return 0

    return dot_product / (norm_s1 * norm_s2)

def predictRatingCosine(user, item):
    item=int(item)
    user=int(user)
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        print(d)
        i2 = d['recipe_id']
        if int(i2) == item:
            ratings.append(0)
        ratings.append(int(d['rating']))
        # print(matrix.getrow(i2))
        # print(matrix.getrow(item))
        print(ratings)
        print("I2",i2)
        print("Item",item)
        similarities.append(Cosine(matrix.getrow(i2), matrix.getrow(item)))
        print(similarities)
    return similarities


#cfPredictions = predictRatingCosine(38094,38094)
cfPredictions = [predictRatingCosine(d['user_id'], int(d['recipe_id'])) for d in dataset]
print(cfPredictions)