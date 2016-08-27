#Similar Movies using public data from MovieLens.org
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
#Take the first 3 cols in the u.data file and import it into the new data frame "r_col" each being sperated by tab
ratings = pd.read_csv('/Users/DanLam/Dropbox/ridentcode/DataScience/DataScience/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3))
# Store the movie_id with it's title
m_cols = ['movie_id', 'title']
movies = pd.read_csv('/Users/DanLam/Dropbox/ridentcode/DataScience/DataScience/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2))

ratings = pd.merge(movies, ratings)

#Merge the 2 created columns using pandas
ratings = pd.merge(movies, ratings)

#Modify our current ratings set: Create an entire matrix of every user and their rating for every movie using the pivot_table
movieRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='ratings')
movieRatings.head()

#Let us test our data on star wars:
starWarsRatings = movieRatings['Star Wars (1977)']
starWarsRatings.head()

#Using pandas we can compute the correlation of each movie with star wars:
similarMovies = movieRatings.corrwith(starWarsRatings)

#Drop the missing results
similarMovives = similarMovies.dropna() 
df = pd.DataFrame
df.head(10)


similarMovies.sort_values(ascending=False)

#BUG: If a user likes an obscure movie and then likes a movie that we are correlating with like Starwars,
# Then that obscure movie would apepar on top of our "similarMovies" list
# However, that truly only benefits the individual user and not the rest of the world
# SOLUTION:
# Remove the movies that haven't been rated by many people
# use pandas to create a table that has the number of ratings and average rating by all users
import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()
#Now we will remove any movie that has lower than 50 ratings
popularMovies = movieStats['rating']['size'] >= 100
#This will basically be the top movies of that time:
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

#Now we will add the popular movie list to our star-wars similarity rating from before:
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
# df.head()

df.sort_values(['similarity'], ascending=False)[:15]
