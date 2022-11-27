import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from pandasql import sqldf
from scipy.spatial.distance import cdist

st.markdown("<h1 style='text-align: center; color: lightblue;'>Spotify Recommendation System</h1>", unsafe_allow_html=True)

# Load the dataset
df = pd.read_csv(r"C:\Users\Admin\econ312prj\features.csv")

#changing release_date to just year only
df['year'] = pd.DatetimeIndex(df['release_date']).year


#move songname, popularity and release year ahead in the df
df.insert(0, 'song_name', df.pop('song_name'))
df.insert(1, 'artist', df.pop('artist'))
df.insert(2, 'year', df.pop('year'))
df.insert(3, 'track_pop', df.pop('track_pop'))

#drop unnecessary columns from the df
drop_columns = ['type', 'id', 'uri', 'track_href', 'analysis_url','release_date']
df.drop(drop_columns, axis = 1, inplace = True)

#replacing < 20 values with mean value
dfmean = df['track_pop'].mean()
df.loc[df['track_pop'] <25, 'track_pop'] = (dfmean + df['track_pop'] + df['artist_pop'])/2
print(df['track_pop'].mean())
df['track_pop'].value_counts()

# new data frame with split value columns of genres
new = df["genres"].str.split(" ", n = 10, expand = True)

# making separate first name column from new data frame
df["first"]= new[0]

# making separate last name column from new data frame
df["second"]= new[1]

# making separate last name column from new data frame
df["third"]= new[2]

# making separate last name column from new data frame
df["fourth"]= new[3]

# making separate last name column from new data frame
df["fifth"]= new[4]

# making separate last name column from new data frame
df["six"]= new[5]

# making separate last name column from new data frame
df["seven"]= new[6]

# making separate last name column from new data frame
df["eight"]= new[7]

# making separate last name column from new data frame
df["nine"]= new[8]

# making separate last name column from new data frame
df["ten"]= new[9]

# making separate last name column from new data frame
df["eleven"]= new[10]

# Dropping old Name columns
df.drop(columns =['genres'], inplace = True)

# Year Selection
start_year, end_year = st.select_slider(
    'Select a range of year',
    options=[i for i in range(2010, 2020)],
    value=(2010, 2011))
st.write('You selected years between', start_year, 'and', end_year)

# Artist Selection
title = st.text_input('Choose your favorite artist')
name = title
title = title.lower()
st.write('You selected', name)

artist = sqldf('select distinct artist from df')
artist_list = artist["artist"].tolist()
artist_list = [x.lower() for x in artist_list]

#Inserting data and all musics fitting to 11 elements of the dataset
df0 = df.copy()
tab1, tab2 = st.tabs(["Surprise", "🗃 Data"])

with tab1:
    tab1.subheader("A tab with surprise")
    df0.dropna(inplace = True)
    st.write(df0)

#Filter out data with selected range of year
with tab2:
    tab2.subheader("A tab with the data")
    df = df.loc[(df['year'] >= start_year)
                     & (df['year'] <= end_year)]
    strings_with_substring = [string for string in artist_list if title in string]
    strings_with_substring = [x.title() for x in strings_with_substring]
    if name == '':
        x = df
        tab2.write(x)
    else:
        for i in strings_with_substring:
            x = df.loc[(df['artist'] == i)]
            if x.empty:
                pass
            else:
                tab2.write(x)


# Recommendation system based on mood
mood_option = st.selectbox(
    'Choose your mood',
    ('Happy', 'Sad', 'Lonely','Hopeful'))

if mood_option == 'Happy':
    happy = sqldf('select distinct song_name, artist, year from x where valence > 0.5 order by track_pop desc limit 10')
    st.write(happy)

if mood_option == 'Sad':
    sad = sqldf('select distinct song_name, artist, year from x where valence < 0.5 order by track_pop desc limit 10')
    st.write(sad)

if mood_option == 'Lonely':
    lonely = sqldf('select distinct song_name, artist, year from x where valence < 0.5 order by acousticness desc, track_pop desc limit 10')
    st.write(lonely)

if mood_option == 'Hopeful':
    hopeful = sqldf('select distinct song_name, artist, year from x where valence > 0.5 order by energy desc, track_pop desc limit 10')
    st.write(hopeful)

# Recommendation system based on genre
#genre = sqldf('SELECT first FROM new UNION SELECT second FROM new UNION SELECT third FROM new UNION SELECT fourth FROM new UNION SELECT fifth FROM new UNION SELECT six FROM new UNION SELECT seven FROM new UNION SELECT eight FROM new UNION SELECT nine FROM new UNION SELECT ten FROM new UNION SELECT eleven FROM new') 
genre_option = st.selectbox(
    'Choose your genre',
    ('atl_hip_hop', 'complextro', 'dance_pop', 'edm', 'electro_house', 'electropop', 'german_techno', 'hip_pop', 'neo_soul', 'pop', 'pop_dance', 'pop_r&b', 'pop_rap', 'post-teen_pop', 'r&b', 'rap', 'tropical_house', 'urban_contemporary'))

if genre_option == 'atl_hip_hop':
    atl_hiphop = sqldf("select distinct song_name, artist, year from x where first like 'atl_hip_hop' or second like 'atl_hip_hop' or third like 'atl_hip_hop' or fourth like 'atl_hip_hop' or fifth like 'atl_hip_hop' or six like 'atl_hip_hop' or seven like 'atl_hip_hop' or eight like 'atl_hip_hop' or nine like 'atl_hip_hop' or ten like 'atl_hip_hop' or eleven like 'atl_hip_hop' order by track_pop desc limit 10")
    st.write(atl_hiphop)
    
if genre_option == 'complextro':
    complextro = sqldf("select distinct song_name, artist, year from x where first like 'complextro' or second like 'complextro' or third like 'complextro' or fourth like 'complextro' or fifth like 'complextro' or six like 'complextro' or seven like 'complextro' or eight like 'complextro' or nine like 'complextro' or ten like 'complextro' or eleven like 'complextro' order by track_pop desc limit 10")
    st.write(complextro)

if genre_option == 'dance_pop':
    dance_pop = sqldf("select distinct song_name, artist, year from x where first like 'dance_pop' or second like 'dance_pop' or third like 'dance_pop' or fourth like 'dance_pop' or fifth like 'dance_pop' or six like 'dance_pop' or seven like 'dance_pop' or eight like 'dance_pop' or nine like 'dance_pop' or ten like 'dance_pop' or eleven like 'dance_pop' order by track_pop desc limit 10")
    st.write(dance_pop)

if genre_option == 'edm':
    edm = sqldf("select distinct song_name, artist, year from x where first like 'edm' or second like 'edm' or third like 'edm' or fourth like 'edm' or fifth like 'edm' or six like 'edm' or seven like 'edm' or eight like 'edm' or nine like 'edm' or ten like 'edm' or eleven like 'edm' order by track_pop desc limit 10")
    st.write(edm)

if genre_option == 'electro_house':
    electro_house = sqldf("select distinct song_name, artist, year from x where first like 'electro_house' or second like 'electro_house' or third like 'electro_house' or fourth like 'electro_house' or fifth like 'electro_house' or six like 'electro_house' or seven like 'electro_house' or eight like 'electro_house' or nine like 'electro_house' or ten like 'electro_house' or eleven like 'electro_house' order by track_pop desc limit 10")
    st.write(electro_house)

if genre_option == 'electropop':
    electropop = sqldf("select distinct song_name, artist, year from x where first like 'electropop' or second like 'electropop' or third like 'electropop' or fourth like 'electropop' or fifth like 'electropop' or six like 'electropop' or seven like 'electropop' or eight like 'electropop' or nine like 'electropop' or ten like 'electropop' or eleven like 'electropop' order by track_pop desc limit 10")
    st.write(electropop)

if genre_option == 'german_techno':
    german_techno = sqldf("select distinct song_name, artist, year from x where first like 'german_techno' or second like 'german_techno' or third like 'german_techno' or fourth like 'german_techno' or fifth like 'german_techno' or six like 'german_techno' or seven like 'german_techno' or eight like 'german_techno' or nine like 'german_techno' or ten like 'german_techno' or eleven like 'german_techno' order by track_pop desc limit 10")
    st.write(german_techno)

if genre_option == 'hip_pop':
    hip_pop = sqldf("select distinct song_name, artist, year from x where first like 'hip_pop' or second like 'hip_pop' or third like 'hip_pop' or fourth like 'hip_pop' or fifth like 'hip_pop' or six like 'hip_pop' or seven like 'hip_pop' or eight like 'hip_pop' or nine like 'hip_pop' or ten like 'hip_pop' or eleven like 'hip_pop' order by track_pop desc limit 10")
    st.write(hip_pop)

if genre_option == 'neo_soul':
    neo_soul = sqldf("select distinct song_name, artist, year from x where first like 'neo_soul' or second like 'neo_soul' or third like 'neo_soul' or fourth like 'neo_soul' or fifth like 'neo_soul' or six like 'neo_soul' or seven like 'neo_soul' or eight like 'neo_soul' or nine like 'neo_soul' or ten like 'neo_soul' or eleven like 'neo_soul' order by track_pop desc limit 10")
    st.write(neo_soul)

if genre_option == 'pop':
    pop = sqldf("select distinct song_name, artist, year from x where first like 'pop' or second like 'pop' or third like 'pop' or fourth like 'pop' or fifth like 'pop' or six like 'pop' or seven like 'pop' or eight like 'pop' or nine like 'pop' or ten like 'pop' or eleven like 'pop' order by track_pop desc limit 10")
    st.write(pop)

if genre_option == 'pop_dance':
    pop_dance = sqldf("select distinct song_name, artist, year from x where first like 'pop_dance' or second like 'pop_dance' or third like 'pop_dance' or fourth like 'pop_dance' or fifth like 'pop_dance' or six like 'pop_dance' or seven like 'pop_dance' or eight like 'pop_dance' or nine like 'pop_dance' or ten like 'pop_dance' or eleven like 'pop_dance' order by track_pop desc limit 10")
    st.write(pop_dance)

if genre_option == 'pop_r&b':
    pop_rb = sqldf("select distinct song_name, artist, year from x where first like 'pop_r&b' or second like 'pop_r&b' or third like 'pop_r&b' or fourth like 'pop_r&b' or fifth like 'pop_r&b' or six like 'pop_r&b' or seven like 'pop_r&b' or eight like 'pop_r&b' or nine like 'pop_r&b' or ten like 'pop_r&b' or eleven like 'pop_r&b' order by track_pop desc limit 10")
    st.write(pop_rb)

if genre_option == 'pop_rap':
    pop_rap = sqldf("select distinct song_name, artist, year from x where first like 'pop_rap' or second like 'pop_rap' or third like 'pop_rap' or fourth like 'pop_rap' or fifth like 'pop_rap' or six like 'pop_rap' or seven like 'pop_rap' or eight like 'pop_rap' or nine like 'pop_rap' or ten like 'pop_rap' or eleven like 'pop_rap' order by track_pop desc limit 10")
    st.write(pop_rap)

if genre_option == 'post-teen_pop':
    post_teen_pop = sqldf("select distinct song_name, artist, year from x where first like 'post-teen_pop' or second like 'post-teen_pop' or third like 'post-teen_pop' or fourth like 'post-teen_pop' or fifth like 'post-teen_pop' or six like 'post-teen_pop' or seven like 'post-teen_pop' or eight like 'post-teen_pop' or nine like 'post-teen_pop' or ten like 'post-teen_pop' or eleven like 'post-teen_pop' order by track_pop desc limit 10")
    st.write(post_teen_pop)

if genre_option == 'r&b':
    rb = sqldf("select distinct song_name, artist, year from x where first like 'r&b' or second like 'r&b' or third like 'r&b' or fourth like 'r&b' or fifth like 'r&b' or six like 'r&b' or seven like 'r&b' or eight like 'r&b' or nine like 'r&b' or ten like 'r&b' or eleven like 'r&b' order by track_pop desc limit 10")
    st.write(rb)

if genre_option == 'rap':
    rap = sqldf("select distinct song_name, artist, year from x where first like 'rap' or second like 'rap' or third like 'rap' or fourth like 'rap' or fifth like 'rap' or six like 'rap' or seven like 'rap' or eight like 'rap' or nine like 'rap' or ten like 'rap' or eleven like 'rap' order by track_pop desc limit 10")
    st.write(rap)

if genre_option == 'tropical_house':
    tropical_house = sqldf("select distinct song_name, artist, year from x where first like 'tropical_house' or second like 'tropical_house' or third like 'tropical_house' or fourth like 'tropical_house' or fifth like 'tropical_house' or six like 'tropical_house' or seven like 'tropical_house' or eight like 'tropical_house' or nine like 'tropical_house' or ten like 'tropical_house' or eleven like 'tropical_house' order by track_pop desc limit 10")
    st.write(tropical_house)

if genre_option == 'urban_contemporary':
    urban_contemporary = sqldf("select distinct song_name, artist, year from x where first like 'urban_contemporary' or second like 'urban_contemporary' or third like 'urban_contemporary' or fourth like 'urban_contemporary' or fifth like 'urban_contemporary' or six like 'urban_contemporary' or seven like 'urban_contemporary' or eight like 'urban_contemporary' or nine like 'urban_contemporary' or ten like 'urban_contemporary' or eleven like 'urban_contemporary' order by track_pop desc limit 10")
    st.write(urban_contemporary)






