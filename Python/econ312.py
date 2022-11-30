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

#page layout and structure
st.set_page_config(page_title="Tobibui1904",layout="wide")
header=st.container()
table = st.container()
analysis = st.container()

#Project Introduction
with header:
    st.markdown("<h1 style='text-align: center; color: lightblue;'>ECON312 Final Project</h1>", unsafe_allow_html=True)
    st.caption("<h1 style='text-align: center;'>Spotify Recommendation System </h1>",unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: red;'>Introduction about the project</h1>", unsafe_allow_html=True)
    st.subheader('1: Project Purposes')
    st.markdown("""The objective of our project is to visualize data on the songs from the Billboard Hot 100 list between 2010 and 2020. We used charts, graphs, scatterplots, and a correlation matrix to track the popularity of songs by genre, artist, and other variables defined below. These visualizations will show us the correlations between variables that would predict the perfect “recipe” for a song to be featured on the Billboard Hot 100 List between 2010 and 2020. 
""")
    st.markdown("""- Danceability: describes how suitable a track is for dancing based on a combination of musical elements, including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is the least danceable, and 1.0 is the most danceable.""")
    st.markdown("""- Energy:  a measure from 0.0 to 1.0 represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale.
""")
    st.markdown("""- instrumentals: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentals value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
""")
    st.markdown("""- Speechiness: detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g., talk show, audiobook, poetry), the closer to 1.0 the attribute value. 
""")
    st.markdown("""- Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g., happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g., sad, depressed, angry) 
""")
    st.subheader('2: Project Team Members')
    left,middle,right=st.columns(3)
        
    st.write("---")

#Display table crawled from Spotify API
with table:
    df = pd.read_csv(r"C:\Users\Admin\econ312prj\features.csv")
    
    #changing release_date to just year only
    df['year'] = pd.DatetimeIndex(df['release_date']).year
    df0 = df.copy()
    
    #move songname, popularity and release year ahead in the df
    df.insert(0, 'song_name', df.pop('song_name'))
    df.insert(1, 'artist', df.pop('artist'))
    df.insert(2, 'year', df.pop('year'))
    df.insert(3, 'track_pop', df.pop('track_pop'))
    
    #drop unnecessary columns from the df
    drop_columns = ['type', 'id', 'uri', 'track_href', 'analysis_url','release_date']
    df.drop(drop_columns, axis = 1, inplace = True)
    
    #string concatnation to list of genre
    df['genres'] = df['genres'].apply(lambda x: x.split(" "))
    
    #replacing < 20 values with mean value
    dfmean = df['track_pop'].mean()
    df.loc[df['track_pop'] <25, 'track_pop'] = (dfmean + df['track_pop'] + df['artist_pop'])/2
    print(df['track_pop'].mean())
    df['track_pop'].value_counts()
    
    #Display table
    st.markdown("<h1 style='text-align: left; color: red;'>Table Summary</h1>", unsafe_allow_html=True)
    st.write(df)
    
    #Display table information
    st.markdown("<h1 style='text-align: center; color: black;'>Table Information</h1>", unsafe_allow_html=True)
    left1, right1 = st.columns(2)
    with left1:
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    
    with right1:
        df1 = df.describe().transpose() 
        st.write(df1)
    st.write("---")

#Analysis about the table
with analysis:
    st.markdown("<h1 style='text-align: left; color: red;'>Table Analysis</h1>", unsafe_allow_html=True)
    
    # Top 30 most occurrence artists from 2010 to 2019
    st.subheader('1: Most Famous Artists')
    fig = plt.figure(figsize=(10,10))
    sns.countplot(y='artist', data=df, order=df['artist'].value_counts().head(30).index).set_title('Top 30 most occurrence artists from 2010 to 2019')
    st.pyplot(fig)
    
    #Comment about the graph
    st.write('We can see that Drake topped the chart with 26 appearances following by Rihanna with 21 appearances.')
    
    # Top 10 genre 2010 - 2019
    fig = plt.figure(figsize=(15,8))
    dfx = pd.Series(sum([item for item in df.genres], [])).value_counts()
    dfx = dfx.to_frame()
    dfx = dfx.reset_index(level=0)
    dfx.columns = ['genre', 'count']
    dfx = dfx.head(10)
    plt.bar(dfx['genre'], dfx['count'], color = ['r','b','yellow','pink','#AFF0f0'])
    plt.title("Top 10 favourite genre from 2010-2019")
    st.write(fig)
    
    #Comment about the graph
    st.write("Pop showed the dominance with the first two positions on the chart (pop, dance_pop) following by rap")

    # Top key used
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig = plt.figure(figsize=(11,8))
    df['key'] = df['key'].replace(regex={0: 'C', 1 : 'C♯/D♭', 2: 'D', 3 : 'D♯/E♭', 4 : 'E', 5 : 'F', 6 : 'F♯/G♭', 7 : 'G', 8 : 'G♯/A♭', 9 : 'A', 10 : 'A♯/B♭', 11 : 'B' })
    sns.countplot(x='key', data=df, order=df['key'].value_counts().index).set_title('Most keys used', fontname="Arial")
    st.write(fig)
    
    #Comment about the graph
    st.write('The most key used was C♯/D♭ (140 appearances) The least used was D♯/E♭ ( <30 appearances)')
    
    # Time Signature Distribution
    fig = plt.figure(figsize=(9,7))
    plt.pie(x = df['time_signature'].value_counts(), labels = df['time_signature'].unique(), autopct='%1.1f%%')
    plt.title('Distribution of time signature', fontsize=15)
    st.write(fig)
    
    #Comment about the graph
    st.write('4/4 is the dominant time seires used (4 beats in each bar) with 95.2%')
    
    # Mode Distribution
    fig = plt.figure(figsize=(9,7))
    plt.pie(x = df['mode'].value_counts(), labels = df['mode'].unique(), autopct='%1.1f%%')
    plt.title('Distribution of mode signature', fontsize=15)
    st.write(fig)
    
    #Comment about the graph
    st.write('dmm')
    
    #getting correlation matrix of musical elements to the listening trend
    f, ax = plt.subplots(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, annot = True, mask=np.zeros_like(corr, dtype=np.bool_), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax).set(title = 'Correlation matrix')
    st.write(f)
    
    #scatterplot for duration and popularity
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(
        df["track_pop"],
        df["duration_ms"],
    )

    ax.set_xlabel("Popularity")
    ax.set_ylabel("Duration")
    
    st.write(fig)
    
    #Comment about the graph
    st.write('dmm')
    
    # Correlation between danceability and song mood
    x = df["danceability"].values
    y = df["valence"].values

    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle("Correlation between danceability and song mood")

    ax = plt.subplot(1, 1, 1)
    ax.scatter(x, y, alpha=0.5)
    ax.plot(x, regr.predict(x), color="red", linewidth=3)
    plt.xticks(())
    plt.yticks(())

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    plt.xlabel("danceability")
    plt.ylabel("valence")

    st.write(fig)
    
    #Comment about the graph
    st.write('dmm')
    
    # 3D graph of relation among popularity, energy and tempo
    plt.rcParams["figure.figsize"] = [8, 8]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title("Correlation betwen tempo, energy and popularity")
    ax.set_xlabel("POPULARITY", fontsize = 12, color = "r")
    ax.set_ylabel("ENERGY", fontsize = 12, color = "r")
    ax.set_zlabel("TEMPO", fontsize = 12, color = "r")
    ax.scatter3D(df['track_pop'], df['energy'], df['tempo'], alpha=1, s=60, color = "b")
    st.write(fig)
    
    #Comment about the graph
    st.write('dmm')
    
    # Subtable with average score of every column in the dataset
    dfx = sqldf('SELECT year, AVG(danceability), AVG(energy), AVG(valence), AVG(loudness), AVG(speechiness), AVG(acousticness), AVG(instrumentalness), AVG(liveness), AVG(tempo), AVG(duration_ms) FROM df0 GROUP BY year')
    dfx = dfx.drop(12)
    
    # Corelation among danceability, energy and song mood by years
    fig = plt.figure(figsize=(8,6))
    plt.title("Danceability and energy and song mood by years")
    plt.plot(dfx['year'], dfx['AVG(danceability)'], label = 'danceability')
    plt.plot(dfx['year'], dfx['AVG(energy)'], label = 'energy')
    plt.plot(dfx['year'], dfx['AVG(valence)'], label = 'valence')
    plt.legend()
    st.write(fig)
    
    #Comment about the graph
    st.write('dmm')
    
    # Corelation among speechiness, aucosticness and liveness by years
    fig = plt.figure(figsize=(8,6))
    plt.title("Speechiness and Aucosticness and Liveness mood by years")
    plt.plot(dfx['year'], dfx['AVG(speechiness)'], label = 'speechiness')
    plt.plot(dfx['year'], dfx['AVG(acousticness)'], label = 'acousticness')
    plt.plot(dfx['year'], dfx['AVG(liveness)'], label = 'liveness')
    plt.legend()
    st.write(fig)
    
    #Comment about the graph
    st.write('dmm')