import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load the datasets
df_ratings = pd.read_csv('rating.csv', delimiter=',').head(200000)
df_anime = pd.read_csv('anime.csv')

# preprocess the data
selected_user_id = 234
df_ratings = df_ratings[df_ratings['rating'] != -1]

# calculate user similarities
user_ratings = df_ratings.pivot_table(
    index='user_id', columns='anime_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_ratings)

# identify similar users
similar_users_indices = user_similarity[selected_user_id-1].argsort()[::-1][1:]

# generate anime recommendations
anime_recommendations = []
for user_index in similar_users_indices:
    similar_user_anime = df_ratings[df_ratings['user_id']
                                    == user_index+1]['anime_id']
    unseen_anime = similar_user_anime[~similar_user_anime.isin(
        df_ratings[df_ratings['user_id'] == selected_user_id]['anime_id'])]
    anime_recommendations.extend(unseen_anime)

# sort and suggest the best 10 anime
recommended_anime_ids = pd.Series(
    anime_recommendations).value_counts().head(10).index

# get the names of recommended anime
recommended_anime_names = df_anime[df_anime['anime_id'].isin(
    recommended_anime_ids)]['name']

print('Top 10 Anime Recommendations: ')
for anime_name in recommended_anime_names:
    print(anime_name)
