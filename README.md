
# What factors significantly impact and determine a movie's success?

##### **Laura Anyanwu, Claire Lee, and Su'ad Mohamud**

## **Introduction**

According to [“How Will the Movies Survive the Next 10 Years?”](https://www.nytimes.com/interactive/2019/06/20/movies/movie-industry-future.html),  the movie industry is beginning to see a lot of changes that affect the profitability and “success” of a film. When major Hollywood figures were questioned about the future of movies, it brought on a discussion about  

*   What drives individuals to the movie theatres? 
*   What makes certain movies worthy of watching at the movie theatres? (“theatricality”) 

In this tutorial, we went in with the assumption that profit generated from a film will be the determining factor of a film's success. We seek to analyze and interpret movie data from 1946 to 2015 to see what factors contributed to the profit of a given film. We look at factors such as gross, budget, IMDb scores, movie directors, and content-rating in order to determine a film’s success. We hope to be able to determine what factors of a film contribute to it's "success" and  how those factors help increase a film's "theatricality". 


### Required Libraries 

Throughout this tutorial, we will use Python 3 as well as various libraries such as [pandas](https://pandas.pydata.org/pandas-docs/stable/), [numpy](https://numpy.org/doc/), [matplotlib](https://matplotlib.org/3.1.1/contents.html), [scipy](https://docs.scipy.org/), [sklearn](https://scikit-learn.org/), and more to manipulate and explore our datasets. 


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statsmodels.formula.api as sm
import sklearn as sklearn
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats
```

## Dataset Source

The dataset used to explore the "success" of a film was [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset#movie_metadata.csv) from Kaggle. The dataset contains 5000+ movies that were scrapped from IMDb. 

The dataset included detailed information for each movie. The dataset has twenty-eight data categories that hold information such as title, director, main actors, genres, gross, budget, language,and more for each movie. The column headings are very descriptive, giving the users an exact understanding of the data that each column holds. 

If there are any questions regarding the column headings or the data, additional information can be found [here.](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset#movie_metadata.csv) 


```python
movies = pd.read_csv('movie_metadata.csv')
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_2_name</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>genres</th>
      <th>...</th>
      <th>num_user_for_reviews</th>
      <th>language</th>
      <th>country</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Color</td>
      <td>James Cameron</td>
      <td>723.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>Joel David Moore</td>
      <td>1000.0</td>
      <td>760505847.0</td>
      <td>Action|Adventure|Fantasy|Sci-Fi</td>
      <td>...</td>
      <td>3054.0</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>237000000.0</td>
      <td>2009.0</td>
      <td>936.0</td>
      <td>7.9</td>
      <td>1.78</td>
      <td>33000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Color</td>
      <td>Gore Verbinski</td>
      <td>302.0</td>
      <td>169.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>Orlando Bloom</td>
      <td>40000.0</td>
      <td>309404152.0</td>
      <td>Action|Adventure|Fantasy</td>
      <td>...</td>
      <td>1238.0</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>300000000.0</td>
      <td>2007.0</td>
      <td>5000.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Color</td>
      <td>Sam Mendes</td>
      <td>602.0</td>
      <td>148.0</td>
      <td>0.0</td>
      <td>161.0</td>
      <td>Rory Kinnear</td>
      <td>11000.0</td>
      <td>200074175.0</td>
      <td>Action|Adventure|Thriller</td>
      <td>...</td>
      <td>994.0</td>
      <td>English</td>
      <td>UK</td>
      <td>PG-13</td>
      <td>245000000.0</td>
      <td>2015.0</td>
      <td>393.0</td>
      <td>6.8</td>
      <td>2.35</td>
      <td>85000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Color</td>
      <td>Christopher Nolan</td>
      <td>813.0</td>
      <td>164.0</td>
      <td>22000.0</td>
      <td>23000.0</td>
      <td>Christian Bale</td>
      <td>27000.0</td>
      <td>448130642.0</td>
      <td>Action|Thriller</td>
      <td>...</td>
      <td>2701.0</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2012.0</td>
      <td>23000.0</td>
      <td>8.5</td>
      <td>2.35</td>
      <td>164000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Doug Walker</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>131.0</td>
      <td>NaN</td>
      <td>Rob Walker</td>
      <td>131.0</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>7.1</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



## Modifying the Data 

We decided to remove and filter the data to narrow the variables we focused on. The data was filtered to only include movies from 1945 to 2015. We wanted to look at movies that correspond with the [beginning of advancements](https://historycooperative.org/the-history-of-the-hollywood-movie-industry/) within the film industry that can be comparable to the quality of films of the present day. 

Also, we decided to remove the columns that did not pertain to the focus of our data exploration. These columns included color, number of critics, number of facebook likes for the director, the number of users voted, total like count for cast members, and more.

Included within the dataset were movies that were missing values for columns such as gross, budget, and movie year. These factors were essential in our tutorial as they were used to establish relationships and perform calculations. Any movies that did not include these variables were removed from our dataset.

### Movie Dataset




```python
#Dropped all attributes of the dataset that we are not concerned about
movies = movies.drop(columns = ['color', 'num_critic_for_reviews', 
                                'director_facebook_likes', 
                                'actor_1_facebook_likes', 'num_voted_users', 
                                'cast_total_facebook_likes', 
                                'actor_3_facebook_likes', 
                                'facenumber_in_poster', 'plot_keywords', 
                                'movie_imdb_link', 'num_user_for_reviews', 
                                'actor_2_facebook_likes', 'aspect_ratio', 
                                'movie_facebook_likes'])
# Drop all rows where either the gross, budget, and/or title_year are undefined
movies = movies.dropna(subset = ['gross'])
movies = movies.dropna(subset = ['budget'])
movies = movies.dropna(subset = ['title_year'])
# Sort the dataset by the title years so the oldest movies show up first and 
# the newest last
movies = movies.sort_values(by = ['title_year'])
# Dropping rows where the movie's title years are not within 1945-2015
movies = movies[movies.title_year > 1944]
movies = movies[movies.title_year < 2016]
periods = ['1940-1949', '1950 - 1959', '1960- 1969', '1970 - 1979',
           '1980 - 1989', '1990 - 1999', '2000 - 2009', '2010 - 2019']
movies['decades'] = pd.cut(movies['title_year'], bins=[1939, 1949, 1959, 1969, 
                                                       1979, 1989, 1999, 2009, 
                                                       2019], labels=periods)
movies['scores'] = pd.cut(movies['imdb_score'], bins = 11)
# Created a profit column so we can see how well received the movie was
movies['profit'] = movies['gross'] - movies['budget']
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>director_name</th>
      <th>duration</th>
      <th>actor_2_name</th>
      <th>gross</th>
      <th>genres</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>actor_3_name</th>
      <th>language</th>
      <th>country</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>imdb_score</th>
      <th>decades</th>
      <th>scores</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4238</th>
      <td>William Wyler</td>
      <td>172.0</td>
      <td>Teresa Wright</td>
      <td>23650000.0</td>
      <td>Drama|Romance|War</td>
      <td>Myrna Loy</td>
      <td>The Best Years of Our Lives</td>
      <td>Dana Andrews</td>
      <td>English</td>
      <td>USA</td>
      <td>Not Rated</td>
      <td>2100000.0</td>
      <td>1946.0</td>
      <td>8.1</td>
      <td>1940-1949</td>
      <td>(7.9, 8.6]</td>
      <td>21550000.0</td>
    </tr>
    <tr>
      <th>3596</th>
      <td>King Vidor</td>
      <td>144.0</td>
      <td>Lillian Gish</td>
      <td>20400000.0</td>
      <td>Drama|Romance|Western</td>
      <td>Joseph Cotten</td>
      <td>Duel in the Sun</td>
      <td>Jennifer Jones</td>
      <td>English</td>
      <td>USA</td>
      <td>Unrated</td>
      <td>8000000.0</td>
      <td>1946.0</td>
      <td>6.9</td>
      <td>1940-1949</td>
      <td>(6.5, 7.2]</td>
      <td>12400000.0</td>
    </tr>
    <tr>
      <th>4328</th>
      <td>Orson Welles</td>
      <td>92.0</td>
      <td>Everett Sloane</td>
      <td>7927.0</td>
      <td>Crime|Drama|Film-Noir|Mystery|Thriller</td>
      <td>Rita Hayworth</td>
      <td>The Lady from Shanghai</td>
      <td>Ted de Corsia</td>
      <td>English</td>
      <td>USA</td>
      <td>Not Rated</td>
      <td>2300000.0</td>
      <td>1947.0</td>
      <td>7.7</td>
      <td>1940-1949</td>
      <td>(7.2, 7.9]</td>
      <td>-2292073.0</td>
    </tr>
    <tr>
      <th>3978</th>
      <td>Vincente Minnelli</td>
      <td>102.0</td>
      <td>Reginald Owen</td>
      <td>2956000.0</td>
      <td>Adventure|Comedy|Musical|Romance</td>
      <td>Gladys Cooper</td>
      <td>The Pirate</td>
      <td>Ellen Ross</td>
      <td>English</td>
      <td>USA</td>
      <td>Approved</td>
      <td>3700000.0</td>
      <td>1948.0</td>
      <td>7.1</td>
      <td>1940-1949</td>
      <td>(6.5, 7.2]</td>
      <td>-744000.0</td>
    </tr>
    <tr>
      <th>3974</th>
      <td>George Sidney</td>
      <td>107.0</td>
      <td>Howard Keel</td>
      <td>8000000.0</td>
      <td>Biography|Comedy|Musical|Romance|Western</td>
      <td>Keenan Wynn</td>
      <td>Annie Get Your Gun</td>
      <td>Betty Hutton</td>
      <td>English</td>
      <td>USA</td>
      <td>Passed</td>
      <td>3768785.0</td>
      <td>1950.0</td>
      <td>7.0</td>
      <td>1950 - 1959</td>
      <td>(6.5, 7.2]</td>
      <td>4231215.0</td>
    </tr>
  </tbody>
</table>
</div>



### Director-Movie Count Dataset

To provide more data analysis for the movie dataset, we decided to look at the relationship between directors and their films within the main movie dataset. 

Below is a dataset that includes the number of movies for thirty popular movie directors. We chose these thirty directors based on [IMDb's list of most popular directors](https://www.imdb.com/list/ls052380992/), as well as our own bias. 

As we can see, Steven Spielberg, Woody Allen, and Clint Eastwood directed the most movies. Alfred Hitchcock, Joon-ho Bong, Ryan Coogler, and Ava DuVernay directed the least. However, these factors can be due to the fact we had to trim our dataset to remove films that had no gross, budget, or title year. Not only that, but with directors that were more active in the 50s and 60s like Alfred Hitchcock, it is very possible that some of his movies were cut from our dataset. As for directors like Ava DuVernay and Ryan Coogler, they only began directing in 2012 and 2013, respectively, thus having fewer movies present on the dataset.


```python
# Create a temporary data structure to hold the popular directors
dList = ['Steven Spielberg', 'Christopher Nolan', 'Quentin Tarantino', 
         'Martin Scorsese', 'David Fincher', 'Woody Allen', 'Robert Zemeckis', 
         'Ridley Scott', 'Francis Ford Coppola', 'Clint Eastwood', 
         'Frank Darabont', 'Joel Coen', 'Alfred Hitchcock', 'Sam Mendes', 
         'Danny Boyle', 'James Cameron', 'Ron Howard', 'Tim Burton', 
         'Darren Aronofsky', 'Roman Polanski', 'Spike Lee', 'Ava DuVernay', 
         'Ryan Coogler', 'Joon-ho Bong', 'George Lucas', 'Lee Daniels', 
         'Ang Lee', 'Chris Columbus', 'James Wan', 'Roland Emmerich']

# Create the director count per movie using the movies dataframe 
mc_d = {}
for director in dList: 
  for (i, row) in movies.iterrows():
    if director in row['director_name']:
      mc_d[director] = mc_d.get(director, 0) + 1

# Aesthetic of the dataframe 
mc_d = pd.DataFrame.from_dict(mc_d, orient='index')
mc_d = mc_d.reset_index()
mc_d.columns = ['Director', 'Movie Count']
mc_d = mc_d.sort_values(by = ['Movie Count'])
mc_d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Director</th>
      <th>Movie Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Alfred Hitchcock</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Ava DuVernay</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Joon-ho Bong</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ryan Coogler</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Lee Daniels</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Joel Coen</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Frank Darabont</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Roman Polanski</td>
      <td>4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>George Lucas</td>
      <td>5</td>
    </tr>
    <tr>
      <th>28</th>
      <td>James Wan</td>
      <td>6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Darren Aronofsky</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>James Cameron</td>
      <td>7</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Roland Emmerich</td>
      <td>7</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Ang Lee</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Quentin Tarantino</td>
      <td>8</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sam Mendes</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Christopher Nolan</td>
      <td>8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Danny Boyle</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Francis Ford Coppola</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>David Fincher</td>
      <td>10</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Chris Columbus</td>
      <td>11</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ron Howard</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Robert Zemeckis</td>
      <td>13</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Spike Lee</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Martin Scorsese</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Tim Burton</td>
      <td>16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ridley Scott</td>
      <td>17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clint Eastwood</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Woody Allen</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Steven Spielberg</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



### Director Dataset

This dataset leaves only the rows of the thirty popular directors.


```python
# Create a new dataframe that is sorted by director, but also cut out directors 
# that we are not focusing on
table = movies.sort_values(by = ['director_name'])
directors = table[table.director_name.isin(mc_d['Director'])]
directors.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>director_name</th>
      <th>duration</th>
      <th>actor_2_name</th>
      <th>gross</th>
      <th>genres</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>actor_3_name</th>
      <th>language</th>
      <th>country</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>imdb_score</th>
      <th>decades</th>
      <th>scores</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2242</th>
      <td>Alfred Hitchcock</td>
      <td>108.0</td>
      <td>Vera Miles</td>
      <td>32000000.0</td>
      <td>Horror|Mystery|Thriller</td>
      <td>Janet Leigh</td>
      <td>Psycho</td>
      <td>John Gavin</td>
      <td>English</td>
      <td>USA</td>
      <td>R</td>
      <td>806947.0</td>
      <td>1960.0</td>
      <td>8.5</td>
      <td>1960- 1969</td>
      <td>(7.9, 8.6]</td>
      <td>31193053.0</td>
    </tr>
    <tr>
      <th>1506</th>
      <td>Ang Lee</td>
      <td>148.0</td>
      <td>Jeffrey Dover</td>
      <td>630779.0</td>
      <td>Drama|Romance|War|Western</td>
      <td>Jeremy W. Auman</td>
      <td>Ride with the Devil</td>
      <td>Tobey Maguire</td>
      <td>English</td>
      <td>USA</td>
      <td>R</td>
      <td>35000000.0</td>
      <td>1999.0</td>
      <td>6.8</td>
      <td>1990 - 1999</td>
      <td>(6.5, 7.2]</td>
      <td>-34369221.0</td>
    </tr>
    <tr>
      <th>2442</th>
      <td>Ang Lee</td>
      <td>112.0</td>
      <td>Kate Burton</td>
      <td>7837632.0</td>
      <td>Drama</td>
      <td>Joan Allen</td>
      <td>The Ice Storm</td>
      <td>Henry Czerny</td>
      <td>English</td>
      <td>USA</td>
      <td>R</td>
      <td>18000000.0</td>
      <td>1997.0</td>
      <td>7.5</td>
      <td>1990 - 1999</td>
      <td>(7.2, 7.9]</td>
      <td>-10162368.0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>Ang Lee</td>
      <td>127.0</td>
      <td>Rafe Spall</td>
      <td>124976634.0</td>
      <td>Adventure|Drama|Fantasy</td>
      <td>Suraj Sharma</td>
      <td>Life of Pi</td>
      <td>Tabu</td>
      <td>English</td>
      <td>USA</td>
      <td>PG</td>
      <td>120000000.0</td>
      <td>2012.0</td>
      <td>8.0</td>
      <td>2010 - 2019</td>
      <td>(7.9, 8.6]</td>
      <td>4976634.0</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Ang Lee</td>
      <td>138.0</td>
      <td>Regi Davis</td>
      <td>132122995.0</td>
      <td>Action|Sci-Fi</td>
      <td>Kevin Rankin</td>
      <td>Hulk</td>
      <td>Celia Weston</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>137000000.0</td>
      <td>2003.0</td>
      <td>5.7</td>
      <td>2000 - 2009</td>
      <td>(5.1, 5.8]</td>
      <td>-4877005.0</td>
    </tr>
  </tbody>
</table>
</div>



### Genre-Movie Count Dataset
This dataframe was used to determine the amount of movies that were in each genre within the current movie dataset.



```python
# This is a list of all the genres noted in the movies dataframe 
glist = movies['genres'].unique()
new_list= []

# Separates the list to the individual genres
for line in glist:
  item = line.split('|')
  new_list.extend(item)

# Make the individual genres unique
x = np.array(new_list)
genre_list = np.unique(x).tolist()

mc_genre = {}

# Create the genre count per movie using the movies dataframe 
for genre in genre_list: 
  for (i, row) in movies.iterrows():
    if genre in row['genres']:
      mc_genre[genre] = mc_genre.get(genre, 0) + 1


# Aesthetic of the dataframe 
mc_genre = pd.DataFrame.from_dict(mc_genre, orient='index')
mc_genre = mc_genre.reset_index()
mc_genre.columns = ['Genre', 'Movie Count']
mc_genre
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>Movie Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>934</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adventure</td>
      <td>770</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Animation</td>
      <td>193</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Biography</td>
      <td>239</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Comedy</td>
      <td>1482</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Crime</td>
      <td>710</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Documentary</td>
      <td>67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Drama</td>
      <td>1932</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Family</td>
      <td>439</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Fantasy</td>
      <td>503</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Film-Noir</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>History</td>
      <td>153</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Horror</td>
      <td>388</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Music</td>
      <td>243</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Musical</td>
      <td>97</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mystery</td>
      <td>380</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Romance</td>
      <td>876</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sci-Fi</td>
      <td>483</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Short</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Sport</td>
      <td>150</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Thriller</td>
      <td>1110</td>
    </tr>
    <tr>
      <th>21</th>
      <td>War</td>
      <td>158</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Western</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



### Genre-Profit Dataset
This dataframe is used to determine the profit that a genre made within a given year.


```python
# Create a temporary data structure to hold the movies that are classified as 
# a genre
mov_genre = {}
for genre in genre_list:
  for (i, row) in movies.iterrows():
    movie = row['movie_title']
    if genre in row['genres']:
      mov_genre.setdefault(genre, []).append(movie)

# Function to format a dataframe for a genre that includes the profit per year. 
def genre_dataframe(df, g): 
  for (i, row) in df.iterrows():
      # Get the profit of a movie 
      profit = movies.loc[movies['movie_title'] == df[0][i], 
                          'profit'].values[0]
      # Get the year of the movie
      year = movies.loc[movies['movie_title'] == df[0][i], 
                        'title_year'].values[0]
      # Get the decade value of the movie 
      period = movies.loc[movies['movie_title'] == df[0][i], 
                          'decades'].values[0]

      # Place in the dataframe
      df.at[i, 'Genre'] = g
      df.at[i, 'Profit'] = profit
      df.at[i, 'Year'] = year
      df.at[i, 'Decade'] = period

  #Aesthetic of the dataframe 
  df.columns = ['Movies', 'Genre', 'Profit', 'Year', 'Decade']

  # Group the dataframe by genre and year and sum the profit of that year 
  df = df.groupby(['Genre', 'Year'])[['Profit']].sum()

  return df
```


```python
# Create a dataframe for 'Action' genre 
values = mov_genre.get('Action')
df = pd.DataFrame(values)
action = genre_dataframe(df, 'Action')

# Create a dataframe for 'Adventure' genre
values = mov_genre.get('Adventure')
df = pd.DataFrame(values)
adv = genre_dataframe(df, 'Adventure')

# Create a dataframe for 'Animation' genre
values = mov_genre.get('Animation')
df = pd.DataFrame(values)
ani = genre_dataframe(df, 'Animation')

# Create a dataframe for 'Biography' genre
values = mov_genre.get('Biography')
df = pd.DataFrame(values)
bio = genre_dataframe(df, 'Biography')

# Create a dataframe for 'Comedy' genre
values = mov_genre.get('Comedy')
df = pd.DataFrame(values)
com = genre_dataframe(df, 'Comedy')

# Create a dataframe for 'Crime' genre
values = mov_genre.get('Crime')
df = pd.DataFrame(values)
cri = genre_dataframe(df, 'Crime')

# Create a dataframe for 'Documentary' genre
values = mov_genre.get('Documentary')
df = pd.DataFrame(values)
doc = genre_dataframe(df, 'Documentary')

# Create a dataframe for 'Drama' genre
values = mov_genre.get('Drama')
df = pd.DataFrame(values)
dra = genre_dataframe(df, 'Drama')

# Create a dataframe for 'Family' genre
values = mov_genre.get('Family')
df = pd.DataFrame(values)
fam = genre_dataframe(df, 'Family')

# Create a dataframe for 'Fantasy' genre
values = mov_genre.get('Fantasy')
df = pd.DataFrame(values)
fan = genre_dataframe(df, 'Fantasy')

# Create a dataframe for 'Film-Noir' genre
values = mov_genre.get('Film-Noir')
df = pd.DataFrame(values)
fil = genre_dataframe(df, 'Film-Noir')

# Create a dataframe for 'History' genre
values = mov_genre.get('History')
df = pd.DataFrame(values)
his = genre_dataframe(df, 'History')

# Create a dataframe for 'Horror' genre
values = mov_genre.get('Horror')
df = pd.DataFrame(values)
hor = genre_dataframe(df, 'Horror')

# Create a dataframe for 'Music' genre
values = mov_genre.get('Music')
df = pd.DataFrame(values)
music = genre_dataframe(df, 'Music')

# Create a dataframe for 'Musical' genre
values = mov_genre.get('Musical')
df = pd.DataFrame(values)
musical = genre_dataframe(df, 'Musical')

# Create a dataframe for 'Mystery' genre
values = mov_genre.get('Mystery')
df = pd.DataFrame(values)
myst = genre_dataframe(df,'Mystery')

# Create a dataframe for 'Romance' genre
values = mov_genre.get('Romance')
df = pd.DataFrame(values)
rom = genre_dataframe(df, 'Romance')

# Create a dataframe for 'Sci-Fi' genre
values = mov_genre.get('Sci-Fi')
df = pd.DataFrame(values)
sci = genre_dataframe(df, 'Sci-Fi')

# Create a dataframe for 'Short' genre
values = mov_genre.get('Short')
df = pd.DataFrame(values)
sho = genre_dataframe(df, 'Short')

# Create a dataframe for 'Sport' genre
values = mov_genre.get('Sport')
df = pd.DataFrame(values)
spo = genre_dataframe(df, 'Sport')

# Create a dataframe for 'Thiller' genre
values = mov_genre.get('Thriller')
df = pd.DataFrame(values)
thr = genre_dataframe(df, 'Thriller')

# Create a dataframe for 'War' genre
values = mov_genre.get('War')
df = pd.DataFrame(values)
war = genre_dataframe(df, 'War')

# Create a dataframe for 'Western' genre
values = mov_genre.get('Western')
df = pd.DataFrame(values)
wes = genre_dataframe(df, 'Western')

# Create a large dataframe with all the genres together 
genre_df = pd.concat([action, adv, ani, bio, com, cri, doc, dra, fam, fan, fil, 
                      his, hor, music, musical, myst, rom, sci, sho, spo, thr, 
                      war, wes], axis= 0)
genre_df = genre_df.reset_index()
genre_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>Year</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>1954.0</td>
      <td>-1730939.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Action</td>
      <td>1962.0</td>
      <td>14967035.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Action</td>
      <td>1963.0</td>
      <td>59700000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Action</td>
      <td>1964.0</td>
      <td>51400000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Action</td>
      <td>1965.0</td>
      <td>54600000.0</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis
In this section of our project, we focused on constructing different plots representing relationships between variables. Some of these plots explore the following: 
- Profit directors made across time
- IMDb scores across time for directors
- Content-rating film count across time 
- Content-rating vs. gross, profit
- IMDb score vs. gross
- Profit vs. year
- Movie Count vs. genre



### Director's Profit. vs Year

Below are scatterplots of each director's movie profits versus the year the movie came out. 


```python
values = sorted(directors['director_name'].unique())  

for pt in values:  
    df = directors[directors['director_name'] == pt]
    
    # Set the parameters for X and Y 
    X = df['title_year'].values.reshape(-1,1)
    Y = df['imdb_score'].values.reshape(-1,1)
    
    # Create the linear regression 
    regr = LinearRegression().fit(X, Y)
    
    # Create the prediction of the X value based on the linear regression
    prediction = regr.predict(X)
    
    # Create a color list
    length = len(df)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(length)]
    
    ax = df.plot.scatter(title="Profit v. Year:\n{}".format(pt),
                  x='title_year', 
                  y='profit',
                  color=colors)
    # Aesthetic for the Graph:  Labels
    ax.set_ylabel('Profit')
    ax.set_xlabel('Year')
    plt.plot(X, prediction, color='k', linewidth=.5)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()
```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)



![png](output_26_5.png)



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)



![png](output_26_9.png)



![png](output_26_10.png)



![png](output_26_11.png)



![png](output_26_12.png)



![png](output_26_13.png)



![png](output_26_14.png)



![png](output_26_15.png)



![png](output_26_16.png)



![png](output_26_17.png)



![png](output_26_18.png)



![png](output_26_19.png)



![png](output_26_20.png)



![png](output_26_21.png)



![png](output_26_22.png)



![png](output_26_23.png)



![png](output_26_24.png)



![png](output_26_25.png)



![png](output_26_26.png)



![png](output_26_27.png)



![png](output_26_28.png)



![png](output_26_29.png)


### Director's IMDb Score vs. Year

Below are scatterplots of each director's movie IMDb scores versus the year it came out.


```python
values = sorted(directors['director_name'].unique())  

for pt in values:  
    df = directors[directors['director_name'] == pt]
    
    # Set the parameters for X and Y 
    X = df['title_year'].values.reshape(-1,1)
    Y = df['imdb_score'].values.reshape(-1,1)
    
    # Create the linear regression 
    regr = LinearRegression().fit(X, Y)
    
    # Create the prediction of the X value based on the linear regression
    prediction = regr.predict(X)
    
    # Create a color list
    length = len(df)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(length)]
    
    ax = df.plot.scatter(title="IMDb Score v. Year:\n{}".format(pt),
                  x='title_year', 
                  y='imdb_score',
                  color=colors)
    # Aesthetic for the Graph:  Labels
    ax.set_ylabel('IMDb Score')
    ax.set_xlabel('Year')
    plt.plot(X, prediction, color='k', linewidth=.5)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()
```


![png](output_29_0.png)



![png](output_29_1.png)



![png](output_29_2.png)



![png](output_29_3.png)



![png](output_29_4.png)



![png](output_29_5.png)



![png](output_29_6.png)



![png](output_29_7.png)



![png](output_29_8.png)



![png](output_29_9.png)



![png](output_29_10.png)



![png](output_29_11.png)



![png](output_29_12.png)



![png](output_29_13.png)



![png](output_29_14.png)



![png](output_29_15.png)



![png](output_29_16.png)



![png](output_29_17.png)



![png](output_29_18.png)



![png](output_29_19.png)



![png](output_29_20.png)



![png](output_29_21.png)



![png](output_29_22.png)



![png](output_29_23.png)



![png](output_29_24.png)



![png](output_29_25.png)



![png](output_29_26.png)



![png](output_29_27.png)



![png](output_29_28.png)



![png](output_29_29.png)


### Director's Profit and IMDb Trend Analysis

From these two charts, we are trying to see if there is a correlation between each director's IMDB scores and profit. With seasoned directors like Steven Spielberg and Quentin Tarantino, we see that they both had a general downward trend in both profit and IMDB score. There may be a correlation between being a seasoned director and their trend with these variables.  The movies they are most well-known for were from earlier in their careers. It is likely that after their breakout films, like Tarantino's ***Pulp Fiction*** and Spielberg's ***Jaws***, their audience started having higher expectations for their films. This led to the audience being more disappointed with the movies that were produced afterward. The effect of this is a lower rating on IMDB and being less likely to go to the movie theaters to watch films they directed. However, it is important to note that despite both directors' general trends being negative, both still had average to above-average ratings on their movies, and made high profits from their films. 
 
One seasoned director that did not quite follow the same trend is Spike Lee, who had a downward trend for profits but demonstrated a consistently positive trend in IMDb ratings. This could potentially be because his later movies are targeted towards a specific demographic or audience. Those in this targeted group watched and rated the film highly.

As for the newer directors like Darren Aronofsky and Christopher Nolan, they are bringing fresh perspectives to the table. Since they are newer directors, the public does not necessarily have high expectations for them as there is no proceeding movie to base their opinion. 

A contradiction to this trend is James Wan, who saw an increasing trend in profits, but a downward trend in ratings. This is likely due to him working primarily on horror franchises, such as the ***Saw*** franchise. Moviegoers had no set standards for the first film, so when movie was released, profits and ratings were high. However, as the franchise progressed, the profits still remained high, but ratings became lower as fans now have a set standard for quality and originality.

From this analysis, we can see two outcomes:
1. Seasoned directors tend to have higher profits due to stronger followings and a trend of decreasing IMDb scores (each are still significantly high) due to public's expectations for their films.
2. Newer directors tend to have lower profits (when compared to season directors) due to marketing infrastructure and higher IMDb scores because of their inventiveness, as they do not have many films available for comparison.
  

### IMDb Score vs. Decades

The following graph is used to showcase the range of scores attributed to fi from each decade. As we can see, movies from the 40s, 50s, and 60s all typically have a mean rating of 7 or 8. We believe that this is due to the lack of documentation and IMDb being created in 1990. With the movies being made decades before the site was created, these movies are ones that people remember and revere. They were the ones that we're able to stand the test of time, and thus why they can still be watched and remembered decades later.

Beginning with the 1980s and onwards, the scores are more varied across the decades, meaning that users are rating a film as soon as they have watched a movie. 


```python
# Create the violinplot
IMDbyear = sns.violinplot(data = movies, x = 'decades', y = 'imdb_score', 
                          fliersize = 150)

# Aesthetics for the violinplot
IMDbyear.set_xlabel("Decades")
IMDbyear.set_ylabel("IMDb Score")
IMDbyear.set_title("IMDb Score vs Decades")
plt.setp(IMDbyear.get_xticklabels(), rotation=45)
plt.show()
```


![png](output_34_0.png)


### Average Budget vs. Content Rating

A relationship we explored was between content-rating and budget. We wanted to see if ratings have an influence on the budget for movie productions. 

As we can see below, movies rated G, PG-13, and PG have the highest budget overall. Rated R movies are in 4th place. Rated X movies had the lowest average budget.


```python
# Create a dataframe that is grouped by content-rating, but then find the mean
# of all numeric attributes
agg = movies.groupby('content_rating').mean()
ratings = agg.index.tolist()
budgets = agg.budget
df = pd.DataFrame({'content_rating': ratings,'budget': budgets})
ax = df.plot(kind = 'bar',    # Plot a bar chart
        legend = False,    # Turn the Legend off
        width = 0.75,      # Set bar width as 75% of space available
        figsize = (8, 6),  # Set size of plot in inches
        color = [plt.cm.Paired(np.arange(len(df)))])

# Aesthetics for the bar chart
ax.set_title('Content Rating vs Average Budget')
ax.set_xlabel('Content Rating')
ax.set_ylabel('Budget in Ten Millions')
plt.show()
```


![png](output_37_0.png)


### Content Rating vs. Gross

We can conclude that movies with PG, G, and M ratings tend to have higher gross averages. However, movies rated PG, PG-13, R, and G have the most extreme outliers.  Movies with ratings "Not Rated", "Unrated", "Passed", "GP", and "NC-17" don't appear often. Due to this limitation, their averages are not necessarily representative of their typical gross. 


```python
# Create a boxplot
grossrating = sns.boxplot(data = movies, x = 'content_rating', y = 'gross')

# Aesthetics for the boxplot
grossrating.set_xlabel("Content Rating")
grossrating.set_ylabel("Gross in Hundred Millions")
grossrating.set_title("Content Rating vs Gross")
plt.setp(grossrating.get_xticklabels(), rotation=45)
plt.show()
```


![png](output_40_0.png)


### Content Rating vs. Profit

Below we created two graphs. In the first graph, we see a few extreme outliers. The outliers represent how much profit was lost. The movies that appeared to have done the worst were from the Rated R category. Rated R appears to have the most extreme outliers, with PG-13 in second place. 

The second graph displays the same data without the negative extreme outliers. In the second graph, we see that PG-13 and PG movies had the most extreme positive outliers. These outliers represent the most profitable movies. Interestingly, Rated R and G movies also had many profitable outliers. 

From our graphs, we can conclude that mainstream ratings ('R', 'PG', 'PG-13', 'G') do affect profitability. 
 


```python
# Create a boxplot
grossprofit = sns.boxplot(data = movies, x = 'content_rating', y = 'profit')

# Aesthetics for the boxplot
grossprofit.set_xlabel("Content Rating")
grossprofit.set_ylabel("Profit in Ten Billions")
grossprofit.set_title("Content Rating vs Profit")
plt.setp(grossprofit.get_xticklabels(), rotation=45)
plt.show()
movies2 = movies.copy()

# Create another graph excluding the extreme outliers
indexNames = movies[movies['profit'] < -20000000].index
movies2 = movies2.drop(indexNames)
grossprofit2 = sns.boxplot(data = movies2, x = 'content_rating', y = 'profit')

# Aesthetics for the boxplot
grossprofit2.set_xlabel("Content Rating")
grossprofit2.set_ylabel("Profit in Hundred Millions")
grossprofit2.set_title("Content Rating vs Profit (Without Extreme Outliers)")
plt.setp(grossprofit2.get_xticklabels(), rotation=45)
plt.show()
```


![png](output_43_0.png)



![png](output_43_1.png)


### IMDb Score vs. Gross

Next, we look at the relationship between IMDb scores and gross. From the plot, we see that the highest rated movies have the highest average gross. However, it is important to note that movies with ratings from 5.8 to 8.6 had a large amount of movies with very high grosses, 4 of which beat the highest grossing movie in the 8.6-9.3  rating. However, there appears to be a general trend of the higher the movie rating, the more money it made.


```python
# Create a boxplot
grossscore = sns.boxplot(data = movies, x = 'scores', y = 'gross')

# Aesthetics for the boxplot
grossscore.set_xlabel("IMDb Score")
grossscore.set_ylabel("Gross in Hundred Millions")
grossscore.set_title("IMDb Score vs Gross")
plt.setp(grossscore.get_xticklabels(), rotation=45)
plt.show()
```


![png](output_46_0.png)


### By the Decade: Profit vs. Year 

We wanted to see if there was a relationship between the profit and the year the film came out. This relation tests to see if there is a general trend that can be observed that may give insight to the amount of profit a film can obtain just by being produced in a certain year.

The first chart looks at the year and profit of films made within a specific decade. This gives us not only an understanding of how films faired during a period but how saturated the industry was during that decade.

The second chart displays the overall year and profit of all films.

The extreme negative outliers were removed from both graphs to give a better overview of the data.

As we have observed in other plots above, as time increases, the profit made becomes more positively consistent across the years. The second graph clearly shows how several films stay within a specific profit margin. Aside from the outliers, most films see a profit but some extremely positive outliers show movies doing particularly better with a specific year.


```python
 values = sorted(movies['decades'].unique())  

df2 = movies[movies['profit'] > -20000000]
length2 = len(df2)
colors2 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(length2)]
# Full Year v. Profit Plot 
bx = df2.plot.scatter(title="Profit v. Year",
                  x='title_year', 
                  y='profit', 
                  color=colors2)

bx.set_ylabel('Profit')
bx.set_xlabel('Year')

plt.setp(bx.get_xticklabels(), rotation=45)

for pt in values: 
    df = movies[movies['decades'] == pt]
    df = df[df['profit'] > -20000000]
    
    # Set the parameters for X and Y 
    X = df['title_year'].values.reshape(-1,1)
    Y = df['profit'].values.reshape(-1,1)
    
    # Create the linear regression 
    regr = LinearRegression().fit(X, Y)
    
    # Create the prediction of the X value based on the linear regression
    prediction = regr.predict(X)
    
    # Create a color list
    length = len(df)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(length)]
    
    # Decade Year v. Profit
    ax = df.plot.scatter(title="Profit v. Year:\n{}".format(pt),
                  x='title_year', 
                  y='profit',
                  color=colors)
    # Line of best fit for Decade Plot
    plt.plot(X, prediction, color='k', linewidth=.5)
    
    # Aesthetic for the Graph:  Labels
    ax.set_ylabel('Profit')
    ax.set_xlabel('Year')
    
    plt.setp(ax.get_xticklabels(), rotation=45)
```


![png](output_48_0.png)



![png](output_48_1.png)



![png](output_48_2.png)



![png](output_48_3.png)



![png](output_48_4.png)



![png](output_48_5.png)



![png](output_48_6.png)



![png](output_48_7.png)



![png](output_48_8.png)


### Movie Count vs. Genre

Utilizing the Movie Count-Genre dataset, we plotted the number of movies within the movies dataset based on their genre. In the movies dataset, most of the movies have more than one genre attributed to it. Due to this, many of the movies are counted many times across each genre. We use this to understand how movies are categorized throughout the year and which were the most popular categorization within the movies datasets. 

Based on this visualization, we can see that most movies are categorized as "Drama", followed by categories of "Comedy" and "Action". There are not many movies that are classified as "Short", "Musical", and "Film-Noir" within the dataset. 

This could suggest that the categories that have a few classifications are not popular genres for movies, whereas films that are considered "Action", "Comedy", and "Drama" are very popular. 


```python
length = len(mc_genre)

# Aesthetics for the bar chart
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(length)]

ax = mc_genre.plot.bar(title = 'Genre vs. Movie Count', x = 'Genre', 
                       y = 'Movie Count', color = colors)
ax.get_legend().remove()
ax.set_ylabel('Movie Count')

plt.show()
```


![png](output_51_0.png)


### By the Decade: Profit vs. Budget 

This visualization is used to show the profits made per year separated by decades. 

Before the 1970s, the linear regression line showed that there was a slight correlation that as the budget increased, the profit of the film increased. 

However, that trend changed after the 1970s, where it can be seen that the budget of movies that came out within their respective decades had similar budgets and had varied profits from those budgets. Especially after the 1980s, nearly every movie that came out during their respective decades were within a very close interval in budgets. 

When a film had a large budget outside the norm of others within the decade, the film tends to fail in profits, causing the negative linear regression line. 

This could suggest that a film's budget began to have very little effect on its profitability. 


```python
values = sorted(movies['decades'].unique())  

for pt in values:  
    decade = movies[movies['decades'] == pt]
    
    # Set the parameters for X and Y 
    X = decade['budget'].values.reshape(-1,1)
    Y = decade['profit'].values.reshape(-1,1)
    
    # Create the linear regression 
    regr = LinearRegression().fit(X, Y)
    
    # Create the prediction of the X value based on the linear regression
    prediction = regr.predict(X)
    
    # Create a color list
    length = len(decade)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(length)]
    
    ax = decade.plot.scatter(title="Budget v. Profit:\n{}".format(pt),
                  x='budget', 
                  y='profit',
                  color=colors)
    
    # Aesthetic for the Graph:  Labels
    ax.set_ylabel('Profit')
    ax.set_xlabel('Budget')
    plt.xticks(rotation ='45')
    
    plt.plot(X, prediction, color='k')
    plt.show()
```


![png](output_54_0.png)



![png](output_54_1.png)



![png](output_54_2.png)



![png](output_54_3.png)



![png](output_54_4.png)



![png](output_54_5.png)



![png](output_54_6.png)



![png](output_54_7.png)


## Machine Learning
### Machine Learning: Simple Linear Regression
Below, we constructed a single linear regression model performed on the movie years to determine what linear relationship exists between year and profit. Due to large outliers present in the data, the overall linear regression model shows a drastic, negative trend in profit across time. We chose to make two plots (one for the line of best fit and the other with the scatterplot) to show the model line clearly. 

To get a more fair representation of the data, we must incorporate an interaction term for each genre of film. This will help us understand the relationship between year and profit for each genre. We did this in the next section. 


```python
# Fit the linear regression model
reg = linear_model.LinearRegression()
Xs = np.array(genre_df['Year']).reshape(-1, 1)
ys = genre_df['Profit']

# Fit the model
reg.fit(Xs, ys)

# Calculate the predictions given the year
profits = reg.predict(np.array(genre_df['Year']).reshape(-1, 1))

# Return the coefficients for the year variable and y intercept.
print("The slope of the model is: {}".format(reg.coef_))
print("The y-intercept of the model is: {}".format(reg.intercept_))
```

    The slope of the model is: [-4483234.71265635]
    The y-intercept of the model is: 8909616033.342546



```python
# Place the linear model onto the same plot
plt.plot(genre_df['Year'], profits, color='green')
# Make a scatter plot of the different profit distributions per year
sns.lmplot(x="Year", y="Profit", data=genre_df, fit_reg=False)
# Title 
plt.title('Year vs Profit linear regression model')
plt.show()
```


![png](output_57_0.png)



![png](output_57_1.png)


### Machine Learning: Multiple Linear Regression
Below, we fit an interaction term in relation to the movie genre to differentiate how the profit changed across time. The model above, which included all movies regardless of genre, is not representative of how profit changes in accordance to genre. Therefore, we created a series of linear regression models for each genre to see if there is drastic and slight change.


```python
m2  = genre_df.copy()
# Fit linear regression model with interaction term on profit per year 
# for each genre
model2 = sm.ols(formula='Profit ~ Year * Genre', data=m2).fit()
# Create a column in the dataframe with predicted values using the 
# regression model
m2['Predicted'] = model2.predict(m2)
# Calculate the residuals by finding the difference between the actual and 
# predicted values
m2['Residual'] = m2['Profit'] - m2['Predicted']
print(model2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 Profit   R-squared:                       0.026
    Model:                            OLS   Adj. R-squared:                 -0.026
    Method:                 Least Squares   F-statistic:                    0.4982
    Date:                Mon, 16 Dec 2019   Prob (F-statistic):              0.998
    Time:                        13:38:40   Log-Likelihood:                -19727.
    No. Observations:                 877   AIC:                         3.954e+04
    Df Residuals:                     832   BIC:                         3.976e+04
    Df Model:                          44                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    Intercept                   3.11e+10   2.56e+10      1.216      0.224   -1.91e+10    8.13e+10
    Genre[T.Adventure]        -1.734e+10   3.39e+10     -0.511      0.609   -8.39e+10    4.92e+10
    Genre[T.Animation]        -9.569e+10   8.27e+10     -1.157      0.248   -2.58e+11    6.67e+10
    Genre[T.Biography]        -3.476e+10   3.92e+10     -0.887      0.375   -1.12e+11    4.22e+10
    Genre[T.Comedy]           -4.168e+10   3.58e+10     -1.164      0.245   -1.12e+11    2.86e+10
    Genre[T.Crime]            -2.542e+10   3.67e+10     -0.693      0.488   -9.74e+10    4.66e+10
    Genre[T.Documentary]      -3.269e+10    6.9e+10     -0.474      0.636   -1.68e+11    1.03e+11
    Genre[T.Drama]             -4.84e+09   3.26e+10     -0.148      0.882   -6.88e+10    5.92e+10
    Genre[T.Family]           -4.195e+10   3.99e+10     -1.052      0.293    -1.2e+11    3.63e+10
    Genre[T.Fantasy]          -4.504e+10   4.39e+10     -1.026      0.305   -1.31e+11    4.11e+10
    Genre[T.Film-Noir]         -139.1707    414.291     -0.336      0.737    -952.350     674.008
    Genre[T.History]          -2.738e+10   3.96e+10     -0.691      0.490   -1.05e+11    5.04e+10
    Genre[T.Horror]           -1.904e+09   3.97e+10     -0.048      0.962   -7.98e+10     7.6e+10
    Genre[T.Music]            -3.402e+10   3.54e+10     -0.961      0.337   -1.03e+11    3.54e+10
    Genre[T.Musical]          -3.014e+10   3.68e+10     -0.818      0.414   -1.02e+11    4.22e+10
    Genre[T.Mystery]          -3.683e+10    4.1e+10     -0.899      0.369   -1.17e+11    4.36e+10
    Genre[T.Romance]           -1.45e+10    3.3e+10     -0.439      0.661   -7.93e+10    5.03e+10
    Genre[T.Sci-Fi]            3.697e+10   4.17e+10      0.886      0.376    -4.5e+10    1.19e+11
    Genre[T.Short]            -3.123e+10   2.31e+11     -0.135      0.892   -4.84e+11    4.22e+11
    Genre[T.Sport]            -2.924e+10   5.54e+10     -0.528      0.598   -1.38e+11    7.94e+10
    Genre[T.Thriller]         -7.356e+09    3.5e+10     -0.210      0.834   -7.61e+10    6.14e+10
    Genre[T.War]              -2.024e+10   3.82e+10     -0.530      0.596   -9.52e+10    5.47e+10
    Genre[T.Western]           -2.92e+10   3.74e+10     -0.781      0.435   -1.03e+11    4.42e+10
    Year                       -1.57e+07   1.29e+07     -1.222      0.222   -4.09e+07    9.52e+06
    Year:Genre[T.Adventure]    8.766e+06   1.71e+07      0.514      0.607   -2.47e+07    4.22e+07
    Year:Genre[T.Animation]    4.794e+07   4.13e+07      1.160      0.246   -3.32e+07    1.29e+08
    Year:Genre[T.Biography]    1.756e+07   1.97e+07      0.893      0.372   -2.11e+07    5.62e+07
    Year:Genre[T.Comedy]       2.114e+07    1.8e+07      1.175      0.240   -1.42e+07    5.65e+07
    Year:Genre[T.Crime]        1.286e+07   1.84e+07      0.698      0.486   -2.33e+07     4.9e+07
    Year:Genre[T.Documentary]  1.651e+07   3.45e+07      0.479      0.632   -5.12e+07    8.42e+07
    Year:Genre[T.Drama]        2.417e+06   1.64e+07      0.147      0.883   -2.98e+07    3.46e+07
    Year:Genre[T.Family]       2.126e+07      2e+07      1.063      0.288    -1.8e+07    6.05e+07
    Year:Genre[T.Fantasy]       2.28e+07    2.2e+07      1.035      0.301   -2.05e+07     6.6e+07
    Year:Genre[T.Film-Noir]   -2.717e+05   8.07e+05     -0.337      0.736   -1.86e+06    1.31e+06
    Year:Genre[T.History]      1.381e+07   1.99e+07      0.695      0.488   -2.52e+07    5.29e+07
    Year:Genre[T.Horror]       9.674e+05   1.99e+07      0.049      0.961   -3.82e+07    4.01e+07
    Year:Genre[T.Music]        1.721e+07   1.78e+07      0.967      0.334   -1.77e+07    5.21e+07
    Year:Genre[T.Musical]      1.523e+07   1.85e+07      0.823      0.411   -2.11e+07    5.16e+07
    Year:Genre[T.Mystery]      1.863e+07   2.06e+07      0.906      0.365   -2.17e+07     5.9e+07
    Year:Genre[T.Romance]       7.31e+06   1.66e+07      0.440      0.660   -2.53e+07    3.99e+07
    Year:Genre[T.Sci-Fi]      -1.868e+07    2.1e+07     -0.891      0.373   -5.98e+07    2.25e+07
    Year:Genre[T.Short]        1.577e+07   1.15e+08      0.137      0.891   -2.11e+08    2.42e+08
    Year:Genre[T.Sport]         1.48e+07   2.77e+07      0.534      0.594   -3.96e+07    6.92e+07
    Year:Genre[T.Thriller]     3.694e+06   1.76e+07      0.210      0.834   -3.09e+07    3.83e+07
    Year:Genre[T.War]          1.021e+07   1.92e+07      0.532      0.595   -2.74e+07    4.78e+07
    Year:Genre[T.Western]      1.475e+07   1.88e+07      0.785      0.433   -2.21e+07    5.16e+07
    ==============================================================================
    Omnibus:                     1415.217   Durbin-Watson:                   1.929
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           566415.154
    Skew:                          -9.936   Prob(JB):                         0.00
    Kurtosis:                     125.905   Cond. No.                     1.00e+16
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 3.63e-23. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.


#### Looking at profit trends over time for two genres: (History and Animation) 
In this section, we selected profit distributions for two specific genres to determine what the different trends are for each genre.

The first genre we selected, History, shows a negative, steadily decreasing trend in profits for films across time. We can also see that there is greater spread after the year 2000. 

The second genre, Animation, has a positive, increasing trend in profits for film across time. We also see 3 specific films situated right before 1990, around 1997, and 2004. 


```python
# Historical and Animated Films Datasets 
history_df = m2[m2['Genre'] == 'History']
animation_df = m2[m2['Genre'] == 'Animation']
```


```python
history_df.reset_index()
history_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>Year</th>
      <th>Profit</th>
      <th>Predicted</th>
      <th>Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>422</th>
      <td>History</td>
      <td>1953.0</td>
      <td>31000000.0</td>
      <td>2.842842e+07</td>
      <td>2.571583e+06</td>
    </tr>
    <tr>
      <th>423</th>
      <td>History</td>
      <td>1962.0</td>
      <td>-9000000.0</td>
      <td>1.143771e+07</td>
      <td>-2.043771e+07</td>
    </tr>
    <tr>
      <th>424</th>
      <td>History</td>
      <td>1963.0</td>
      <td>26635000.0</td>
      <td>9.549852e+06</td>
      <td>1.708515e+07</td>
    </tr>
    <tr>
      <th>425</th>
      <td>History</td>
      <td>1965.0</td>
      <td>-12000000.0</td>
      <td>5.774140e+06</td>
      <td>-1.777414e+07</td>
    </tr>
    <tr>
      <th>426</th>
      <td>History</td>
      <td>1970.0</td>
      <td>2200000.0</td>
      <td>-3.665143e+06</td>
      <td>5.865143e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
animation_df.reset_index()
animation_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>Year</th>
      <th>Profit</th>
      <th>Predicted</th>
      <th>Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>106</th>
      <td>Animation</td>
      <td>1988.0</td>
      <td>-1.063968e+09</td>
      <td>-4.970570e+08</td>
      <td>-5.669110e+08</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Animation</td>
      <td>1992.0</td>
      <td>1.893502e+08</td>
      <td>-3.680968e+08</td>
      <td>5.574470e+08</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Animation</td>
      <td>1993.0</td>
      <td>-2.733072e+07</td>
      <td>-3.358568e+08</td>
      <td>3.085260e+08</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Animation</td>
      <td>1994.0</td>
      <td>3.777838e+08</td>
      <td>-3.036167e+08</td>
      <td>6.814005e+08</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Animation</td>
      <td>1995.0</td>
      <td>2.483962e+08</td>
      <td>-2.713767e+08</td>
      <td>5.197729e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Historical film scatter plot 
sns.scatterplot(x=history_df['Year'], y=history_df['Profit'])
# Fit interaction model on history profits
plt.plot(history_df['Year'], history_df['Predicted'], color='purple')
plt.title('Profits over time for Historical Films')
plt.show()
```


![png](output_64_0.png)



```python
# Animated film scatter plot 
sns.scatterplot(x=animation_df['Year'], y=animation_df['Profit'])
# Fit interaction model on animation profits
plt.plot(animation_df['Year'], animation_df['Predicted'], color='green')
plt.title('Profits over time for Animated Films')
plt.show()
```


![png](output_65_0.png)


### Write up: Genre as an Interaction term 
Above, we can see that an interaction term was necessary to understand the trends of profit for our film dataset. We can see that there is a relationship between year and profit for each genre. Additionally, we can see that this relationship is different for each corresponding genre. Above, we determined that historical films have been decreasing in profit across time while animated films have increased steadily across time.

From the table above of our regression results, we can see that the p-values are larger than our significance value of 0.05. Therefore, we fail to reject our null-hypothesis that genre does not determine a film's success. We must accept the null-hypothesis because there is insufficient evidence to support the claim that movie genre determines a film's success. For further exploration, we can look at interaction terms for the IMBD scores and others to determine if they have a significant impact on a film's success.


### Violinplot for Residual Profits by Genre
Below is a residual plot for profits for each genre. We split the genre list in half to get a better look at the violin plots. 

From our plots, we can see that there are certain genres with wider violinplots because of outliers. Some of these include Action, Adventure, Comedy and Science Fiction. This is likely due to the greater population sizes of these genres. In contrast, there are certain genres with smaller violinplots. This is due to a small number of films of that genre type. The genre with the widest violinplot is Sci-Fi. 


```python
# Genres: Action, Adventure, Animation, Biography, Comedy, Crime, Documentary
# Drama, Family, Fantasy, Film-Noir

# Split the profits vs. genre dataframe into two dataframes
m3 = m2[m2['Genre'] < 'H'] 
m4 = m2[m2['Genre'] > 'H'] 
# First half (first 11 genres alphabetically) of residual violinplots 
sns.violinplot(x=m3['Genre'], y=m3['Residual']).set_title('Residual Profits')
plt.xticks(rotation ='45')
plt.show()
```


![png](output_68_0.png)



```python
# History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Short, Sport
# Thriller, War, Western

# Second half (second 12 genres alphabetically) of residual violinplots 
sns.violinplot(x=m4['Genre'], y=m4['Residual']).set_title('Residual Profits')
plt.xticks(rotation ='45')
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]),
     <a list of 12 Text xticklabel objects>)




![png](output_69_1.png)


# Conclusion: 

In summary, we wanted to test and explore a movie dataset to figure out which variables within a film will determine whether or not it becomes successful. To do this, we established that profit was an important factor to analyze and incorporate into our data exploration. We began by cleaning up our data for missing values and removing unnecessary variables to narrow down our search. We selected IMDb scores, profit, genres, director names, gross, budget, and content-ratings as columns for our main dataset. 

We generated a series of dataframes to better understand the distribution of our dataset. This meant tracking down the allocation of films by genre and director. In addition to this, we determine how much profit each genre made over time. From this, we found that Drama, Comedy and Thriller were the most popular genres. For directors, Steven Spielberg, Woody Allen, and Clint Eastwood have directed the highest number of films.

In our exploration section, we took a look at profit distributions for our 30 selected directors across time. Then, we looked at what content-ratings were the most common (we found that PG-13, PG, and G had the highest counts in that order respectively). We determined that profit and budget had a positive relationship before the 70s. This is likely because there were less films being made and available before 1970. Additionally, the profit distribution across time showed that profits were generally positive (with the exception of a few outliers) until the 21 century. In the 21 century plot, we can see that there are more films below and above the x axis, indicating that more films made negative profit (made less than their budget) as well as more films that made more profit. From our IMDb scores vs. gross, we can see that gross and IMDb scores have a positive relationship, so when IMDb scores ranges increase, when gross increases. 

For our machine learning section, we performed both simple and multiple regression on movie title year and profit. At a significance level (alpha) of 0.05, we found that we cannot reject the null hypothesis that year does not determines a film's success. Additionally, we discovered that our interaction term (genre) did not have a significant impact in determining a film's success either. This was determined because our p-values were greater than 0.05. 

Through this project, we were able to learn more about how the characteristics of films affect their profitability.  We were able to interpret why films produced within a specific decade exhibit certain trends (i.e. films made closer to the 40s and 50s had produced higher profit values due to the lack of films available and because films became more viable after this time). We can conclude that there is a  relationship present between an IMDB score and profit (generally IMDb scores are higher for films with higher profit). Also, that characteristics such as director and budget have an effect on the profit of a film. However, we weren't able to determine what characteristic(s) directly affect a film's success. We believe that more factors need to be considered when determining what creates the "theatricality" for a film and what drives people to the movie theatre. We encourage future projects to look deeper into other characteristics of films to determine this answer.

# Project Notes
This project was created by Laura Anyanwu, Claire Lee, and Su'ad Mohamud as a project for the CMSC 320 Data Science class offered to undergraduate students under the Computer Science department at the University of Maryland-College Park. 

# Credits
The data was obtained from [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset#movie_metadata.csv). The visualizations were designed using Python libraries.

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
