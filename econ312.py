import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from PIL import Image
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
    st.markdown("""This project has 2 main parts: Analysis about the general music trend from 2010 to 2019 via Spotify and Build a simple recommendation system based on users' preferences """)
    st.subheader('2: Project Team Members')
    left,middle,right=st.columns(3)
    with left:
        image = Image.open(r"C:\Users\Admin\econ312prj\_ART6463.jpg")
        st.image(image, caption='Tobi Bui: Designing the recommandation system and website')
    with middle:
        image = Image.open(r"C:\Users\Admin\econ312prj\_ART6463.jpg")
        st.image(image, caption='Nguyen Tran: Data Analyst')
    with right:
        image = Image.open(r"C:\Users\Admin\econ312prj\_ART6463.jpg")
        st.image(image, caption='Audrey: Project Manager')
        
    st.write("---")

#Display table crawled from Spotify API
with table:
    df = pd.read_csv(r"C:\Users\Admin\econ312prj\all.csv")
    
    #changing release_date to just year only
    df['year'] = pd.DatetimeIndex(df['release_date']).year
    
    #move songname, popularity and release year ahead in the df
    df.insert(0, 'name', df.pop('name'))
    df.insert(1, 'artist', df.pop('artist'))
    df.insert(2, 'year', df.pop('year'))
    df.insert(3, 'popularity', df.pop('popularity'))
    
    #drop unnecessary columns from the df
    drop_columns = ['type', 'id', 'uri', 'track_href', 'analysis_url','release_date']
    df.drop(drop_columns, axis = 1, inplace = True)
    
    #replacing 0 values with mean value
    df['popularity'] = df['popularity'].replace(0,df['popularity'].mean())
    
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
    st.write('From the graph we can see that Drake has the highest occurrence over a decade with 25 appearance in Spotify list')
    st.write('In the meantime, Florida has the lowest occurrence over a decade with 7 appearance in Spotify list')
    
    # Most Impact Musical Elements on Users'preferences
    st.subheader("2: Most Impact Musical Elements on Users'preferences")
    f, ax = plt.subplots(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, annot = True, mask=np.zeros_like(corr, dtype=np.bool_), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.title('Correlation Matrix of track data')
    st.pyplot(f)
    
    #comment about the graph
    st.write('dmm')
    
    #scatterplot for duration and popularity
    st.subheader("3: Correlation between Duration and Popularity")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(
        df["popularity"],
        df["duration_ms"],
    )

    ax.set_xlabel("Acceleration")
    ax.set_ylabel("Duration")

    st.write(fig)
    
    #comment about the graph
    st.write('dmm')
    
    #Correlation between danceability and song mood
    st.subheader("4: Correlation between Danceability and Song Mood")
    x = df["danceability"].values
    y = df["valence"].values

    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    fig = plt.figure(figsize=(4, 4))

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
    plt.ylabel("Song Mood")

    st.write(fig)
    
    #comment about the graph
    st.write('dmm')
    
    # 3D graph of relation between Popularity, Energy and Tempo
    st.subheader('5: 3D graph of relation between Popularity, Energy and Tempo')
    plt.rcParams["figure.figsize"] = [8, 8]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("POPULARITY", fontsize = 12, color = "r")
    ax.set_ylabel("ENERGY", fontsize = 12, color = "r")
    ax.set_zlabel("TEMPO", fontsize = 12, color = "r")
    ax.scatter3D(df['popularity'], df['energy'], df['tempo'], alpha=1, s=60, color = "b")
    st.write(fig)
    
    #comment about the graph
    st.write('dmm')
    
    # A more general look about correlation in music element via average score
    
    # Creating sub table representing the average score for music elements 
    st.subheader('6: Correlation among music elements on average')
    st.subheader('6.1: Sub table representing the average score for music elements ')
    dfx = sqldf('SELECT year, AVG(popularity), AVG(danceability), AVG(energy), AVG(speechiness), AVG(acousticness), AVG(instrumentalness), AVG(liveness), AVG(valence), AVG(tempo) FROM df GROUP BY year')
    dfx = dfx.drop(12)
    st.write(dfx)
    
    # Correlation between Danceability, energy and song mood by years
    st.subheader("6.2: Danceability and energy and valence by years")
    fig = plt.figure(figsize=(10,8))
    plt.plot(dfx['year'], dfx['AVG(danceability)'], label = 'danceability')
    plt.plot(dfx['year'], dfx['AVG(energy)'], label = 'energy')
    plt.plot(dfx['year'], dfx['AVG(valence)'], label = 'valence')
    plt.legend()
    st.write(fig)
    
    #comment about the graph
    st.write('dmm')
    
    # Correlation between Speechiness, Acousticness and Liveness by years
    st.subheader("6.3: Speechiness and Acousticness and Liveness by years")
    fig = plt.figure(figsize=(8,6))
    plt.plot(dfx['year'], dfx['AVG(speechiness)'], label = 'speechiness')
    plt.plot(dfx['year'], dfx['AVG(acousticness)'], label = 'acousticness')
    plt.plot(dfx['year'], dfx['AVG(liveness)'], label = 'liveness')
    plt.legend()
    st.write(fig)
    
    #comment about the graph
    st.write('dmm')