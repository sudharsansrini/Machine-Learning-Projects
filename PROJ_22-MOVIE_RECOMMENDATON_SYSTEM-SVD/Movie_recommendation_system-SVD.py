import pandas as pd
import numpy as np

ratingData = pd.io.parsers.read_csv('ratings.dat', names=['user_id', 'movie_id','rating', 'time'], engine='python',
                                    delimiter='::')
movieData = pd.io.parsers.read_csv('movies.dat', names=['movie_id', 'title', 'genre'], engine='python',
                                   delimiter='::')

print(ratingData.shape)
print(movieData.shape)
print(ratingData.head(5))
print(movieData.head(5))

ratingMatrix = np.ndarray(shape=(np.max(movieData.movie_id.values), np.max(ratingData.user_id.values)),
                          dtype=np.uint8)

print(ratingMatrix)

