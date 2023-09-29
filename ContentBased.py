import pandas as pd

# load data
anime_data = pd.read_csv(
    '/Users/fatemehsoufian/Downloads/anime/anime.csv')
ratings_data = pd.read_csv(
    '/Users/fatemehsoufian/Downloads/anime/rating.csv')

anime_data.dropna(inplace=True)
ratings_data.dropna(inplace=True)

# split genres
anime_data['genre'] = anime_data.genre.str.split(', ')
anime_with_genres = anime_data.copy(deep=True)

x = []
for index, row in anime_data.iterrows():
    x.append(index)
    for genre in row['genre']:
        anime_with_genres.at[index, genre] = 1
anime_with_genres = anime_with_genres.fillna(0)

# get the user id
user_id = int(input('Enter your id: '))

# get all the ratings user gave
user_ratings = ratings_data[ratings_data['user_id'] == user_id]

# get all anime ids that user rated
user_anime_id = anime_data[anime_data['anime_id'].isin(
    user_ratings['anime_id'])].drop(['rating'], 1)

# merge the user rating and the anime id
user_ratings = pd.merge(user_anime_id, user_ratings)

# drop the unnecessary cols
user_ratings = user_ratings.drop(
    ['genre', 'type', 'episodes', 'members', 'user_id'], 1)

# get the df with anime genres that user watched
user_genres_df = anime_with_genres[anime_with_genres['anime_id'].isin(
    user_ratings['anime_id'])]

# reset the index
user_genres_df.reset_index(drop=True, inplace=True)

# drop the unnecessary cols
user_genres_df.drop(['anime_id', 'name', 'genre', 'type',
                    'episodes', 'rating', 'members'], axis=1, inplace=True)

# give weight based on ratings user gave to the genres
user_profile = user_genres_df.T.dot(user_ratings.rating)

# as there are negative ratings abs them
user_profile = user_profile.abs()

# set anime_id col as index
anime_with_genres = anime_with_genres.set_index(anime_with_genres['anime_id'])

# drop the unnecessary cols
anime_with_genres.drop(['anime_id', 'name', 'genre', 'type',
                       'episodes', 'rating', 'members'], axis=1, inplace=True)

recommendation_table_df = (anime_with_genres.dot(
    user_profile)) / user_profile.sum()

# sort
recommendation_table_df.sort_values(ascending=False, inplace=True)

# remove animes that the user already watched
recommendation_table_df = recommendation_table_df[~recommendation_table_df.index.isin(
    user_ratings['anime_id'])]

# output the top 10
copy = anime_data.copy(deep=True)
copy = copy.set_index('anime_id', drop=True)
top_10_index = recommendation_table_df.index[:10].tolist()
recommended_animes = copy.loc[top_10_index, :]
print(recommended_animes)
