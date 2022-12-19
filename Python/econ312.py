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
from sklearn.metrics import mean_squared_error
import altair as alt

st.set_page_config(page_title="Tobibui1904",layout="wide")

def main_page():
    st.sidebar.markdown("# Visualization")
    
    #page layout and structure
    header=st.container()
    table = st.container()
    analysis = st.container()
    conclusion = st.container()

    #Project Introduction
    with header:
        st.markdown("<h1 style='text-align: center; color: lightblue;'>ECON312 Final Project</h1>", unsafe_allow_html=True)
        st.caption("<h1 style='text-align: center;'>Spotify Recommendation System </h1>",unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: left; color: red;'>Introduction about the project</h1>", unsafe_allow_html=True)
        st.subheader('1: Project Purposes')
        st.markdown("""The objective of our project is to collect and visualize data on the songs from the Billboard Hot
    100 list between 2010 and 2020 and build a simple prediction model using Linear Regression. We will use
    that model to pick out the top 10 songs of 2021 and 2022 with the highest popularity prediction. We also
    build a simple recommendation system based on mood and also a filter for artists, years and genres 
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
        with left:
            st.write("Nguyen Tran")
        with middle:
            st.write("Tobi Bui")
        with right:
            st.write("Audrey Burkey")
            
        st.write("---")

    #Display table crawled from Spotify API
    with table:
        df = pd.read_csv("features.csv")
        
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

        #Converting dataframe columns to Integer
        df["track_pop"] = df["track_pop"].astype(int)
        
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
            
            #Converting dataframe columns to Integer
            df1["count"] = df1["count"].astype(int)
            df1["min"] = df1["min"].astype(int)
            
            st.write(df1)
            
        st.write("---")

    #Analysis about the table
    with analysis:
        st.markdown("<h1 style='text-align: left; color: red;'>Table Analysis</h1>", unsafe_allow_html=True)
        
        # Top 30 most occurrence artists from 2010 to 2019
        st.subheader('1: Most Famous Artists')
        fig = plt.figure(figsize=(4,5))
        sns.countplot(y='artist', data=df, order=df['artist'].value_counts().head(30).index).set_title('Top 30 most occurrence artists from 2010 to 2019')
        st.pyplot(fig)
        
        #Comment about the graph
        st.write('This graph visualizes the number of songs on the Billboard Hot 100 list the top 30 artists had between 2010-2020. Drake leads the charts with more than 25 songs')
        
        # Top 10 genre 2010 - 2019
        st.subheader('2: Most Famous Genre')
        fig = plt.figure(figsize=(15,6))
        dfx = pd.Series(sum([item for item in df.genres], [])).value_counts()
        dfx = dfx.to_frame()
        dfx = dfx.reset_index(level=0)
        dfx.columns = ['genre', 'count']
        dfx = dfx.head(10)
        plt.bar(dfx['genre'], dfx['count'], color = ['r','b','yellow','pink','#AFF0f0'])
        plt.title("Top 10 favourite genre from 2010-2019")
        st.write(fig)
        
        #Comment about the graph
        st.write("Four out the ten top genres of the Billboard Hot 100 lists between 2010 and 2020 include elements of “pop” music, with generic pop music having the most hits. Pop music is defined as music that serves purposes for “light entertainment, commercial imperatives, and personal identification.” These associations with pop music make radio and subsequently the Billboard Hot 100 great environments for pop music to thrive.")

        # Top key used
        st.subheader('3: Distribution of major or minor keys in hit songs between 2010-2020:')
        plt.rcParams.update({'font.family': 'DejaVu Sans'})
        fig = plt.figure(figsize=(11,6))
        df['key'] = df['key'].replace(regex={0: 'C', 1 : 'C♯/D♭', 2: 'D', 3 : 'D♯/E♭', 4 : 'E', 5 : 'F', 6 : 'F♯/G♭', 7 : 'G', 8 : 'G♯/A♭', 9 : 'A', 10 : 'A♯/B♭', 11 : 'B' })
        sns.countplot(x='key', data=df, order=df['key'].value_counts().index).set_title('Most keys used', fontname="Arial")
        st.write(fig)
        
        #Comment about the graph
        st.write('There are more songs in a minor key than a major key within our dataset, showing that songs that are less positive, happy or peppy are more popular. This coincides with the data showing that the valence of hit songs has decreased over time.')
        
        # Time Signature Distribution
        st.subheader('4: Time signature distribution between 2010-2020:')
        fig = plt.figure(figsize=(9,7))
        plt.pie(x = df['time_signature'].value_counts(), labels = df['time_signature'].unique(), autopct='%1.1f%%')
        plt.title('Distribution of time signature', fontsize=15)
        plt.show()
        st.write(fig)
        
        #Comment about the graph
        st.write('95.2% of songs on the Billboard Hot 100 list between 2010 and 2020 used the time signature of 4/4, a simple measure mostly used in pop songs.')
        
        # Mode Distribution
        st.subheader('5: Mode distribution between 2010-2020:')
        fig = plt.figure(figsize=(2,2))
        
        data = [63, 37]
        label = ['0', '1']
        
        plt.pie(data, labels=label, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Mode distribution between 2010-2020')
        plt.axis('equal')
        st.write(fig)
        
        #Comment about the graph
        st.write('The majority of songs are written in minor scale (63%)')
        
        #getting correlation
        st.subheader('6: Correlation Matrix:')
        f, ax = plt.subplots(figsize=(9, 7))
        corr = df.corr()
        sns.heatmap(corr, annot = True, mask=np.zeros_like(corr, dtype=np.bool_), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax).set(title = 'Correlation matrix')
        st.write(f)
        
        # Comment about the graph
        st.write("The correlation matrix displays the correlation values between the variables we are analyzing between these songs. This matrix gives us insight into what factors contribute to a song being on the Billboard Hot 100 list between 2010 and 2020. There is a medium correlation of .36 between an artist’s popularity and their track’s popularity, illustrating that if an artist is already perceived to be popular, then their track will perform better. Most other variables had a weak correlation with a track’s popularity, but “acousticness” and “danceability” both stick out as good qualities to have in a hit song. ")
        
        #scatterplot for duration and popularity
        st.subheader('7: Scatterplot of track duration in milliseconds and track popularity')
        fig = alt.Chart(df).mark_circle(size=60).encode(
            x='track_pop',
            y='duration_ms',
        ).properties(
            width=900,
            height=600
        )
        st.write(fig)
        
        #Comment about the graph
        st.write('There is no clear correlation between track length and track popularity.')
        
        # Correlation between danceability and song mood
        st.subheader('8: Correlation between danceability and valence (song mood)')
        x = df["danceability"].values
        y = df["valence"].values

        x = x.reshape(x.shape[0], 1)
        y = y.reshape(y.shape[0], 1)

        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        fig = plt.figure(figsize=(6, 6))
        fig.suptitle("Correlation between danceability and song mood")

        ax = plt.subplot(2, 1, 1)
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
        st.write('The higher the song mood, the more it is perceived to be danceable.')
        
        # 3D graph of relation among popularity, energy and tempo
        st.subheader('9: Correlation between tempo, energy and popularity')
        plt.rcParams["figure.figsize"] = [7, 7]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("POPULARITY", fontsize = 10, color = "r")
        ax.set_ylabel("ENERGY", fontsize = 10, color = "r")
        ax.set_zlabel("TEMPO", fontsize = 10, color = "r")
        ax.scatter3D(df['track_pop'], df['energy'], df['tempo'], alpha=1, s=20, color = "b")
        st.write(fig)
        
        #Comment about the graph
        st.write('There is a positive correlation between track popularity and track tempo, leaning towards a higher tempo for a more popular song.')
        
        # Subtable with average score of every column in the dataset
        dfx = sqldf('SELECT year, AVG(danceability), AVG(energy), AVG(valence), AVG(loudness), AVG(speechiness), AVG(acousticness), AVG(instrumentalness), AVG(liveness), AVG(tempo), AVG(duration_ms) FROM df0 GROUP BY year')
        dfx = dfx.drop(12)
        
        # Corelation among danceability, energy and song mood by years
        st.subheader('10: Tracking Danceability, energy, and valence by years:')
        fig = plt.figure(figsize=(8,4))
        plt.title("Danceability and energy and song mood by years")
        plt.plot(dfx['year'], dfx['AVG(danceability)'], label = 'danceability')
        plt.plot(dfx['year'], dfx['AVG(energy)'], label = 'energy')
        plt.plot(dfx['year'], dfx['AVG(valence)'], label = 'valence')
        plt.legend()
        st.write(fig)
        
        #Comment about the graph
        st.write('We see an overall incline in the amount of “danceability” a song has on the Billboard Hot 100 between 2010 and 2020. The valence (described as the musical positiveness of a song) of songs has gone through many trends but overall we have seen a decrease in the positiveness of songs between 2010 and 2020. Finally, the energy of songs has decreased, surprisingly, even when danceability of songs has increased.')
        
        # Corelation among speechiness, aucosticness and liveness by years
        st.subheader('11: Tracking speechiness, acousticness, and liveness by year')
        fig = plt.figure(figsize=(8,4))
        plt.title("Speechiness and Aucosticness and Liveness mood by years")
        plt.plot(dfx['year'], dfx['AVG(speechiness)'], label = 'speechiness')
        plt.plot(dfx['year'], dfx['AVG(acousticness)'], label = 'acousticness')
        plt.plot(dfx['year'], dfx['AVG(liveness)'], label = 'liveness')
        plt.legend()
        st.write(fig)
        
        #Comment about the graph
        st.write('Out of the variables shown, we see that speechiness has increased the most between 2008 and 2018, followed by the acousticness of a song. The increases in these variables coincide with the increase in “danceability” of songs, leading to more instrumental-leading hit songs rather than speech or lyric leading hits.')
        
        st.write("---")

    with conclusion:
        st.markdown("<h1 style='text-align: left; color: red;'>Conclusion about the project</h1>", unsafe_allow_html=True)
        st.write('Based on our visualization, we have concluded the top three variables of importance when concocting a song to be featured on the Billboard Hot 100 list between 2010 and 2020: danceability, valence, and speechiness. Danceability and valence seem to go hand in hand, creating an upbeat, happy / positive track. The increase in speechiness can be thought of through the framework of rap and spoken words being called out through a song, again correlating to more danceable songs. In conclusion, our data shows that the most popular songs between 2010 and 2020 were danceable and upbeat, with callouts throughout, making the perfect club and radio hit to be featured on the Billboard Hot 100 list. These popularity trends also correlate to an increase in popularity of video-making apps like Tik Tok; some of the most popular videos on the app are dance videos, possibly giving incentives to artists to create more “danceable” tracks.')


def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Spotify Recommendation System</h1>", unsafe_allow_html=True)
    
    # Load the dataset
    df = pd.read_csv("features.csv")

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

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")
    
    st.markdown("<h1 style='text-align: center; color: lightblue;'>2021 and 2022 Prediction</h1>", unsafe_allow_html=True)

    traindf = pd.read_csv("features.csv")
    predf = pd.read_csv("2021.csv")
    pre22df = pd.read_csv("2022.csv")
    drop_columns = ['type', 'id', 'uri', 'track_href', 'analysis_url','release_date']

    # Update traindf with release_date to year and track_pop to more precise
    st.markdown("<h1 style='text-align: left; color: red;'>Original Table with year updated</h1>", unsafe_allow_html=True)
    traindf['year'] = pd.DatetimeIndex(traindf['release_date']).year
    traindf.drop(drop_columns, axis = 1, inplace = True)
    dfmean = traindf['track_pop'].mean()
    traindf.loc[traindf['track_pop'] <25, 'track_pop'] = (dfmean + traindf['track_pop'] + traindf['artist_pop'])/2
    st.write(traindf.head())

    # Update 2021 data with release_date to year
    st.markdown("<h1 style='text-align: left; color: red;'>2021 Table with year updated</h1>", unsafe_allow_html=True)
    predf['year'] = pd.DatetimeIndex(predf['release_date']).year
    predf.drop(drop_columns, axis = 1, inplace = True)
    st.write(predf.head())

    # Update 2022 data with release_date to year
    st.markdown("<h1 style='text-align: left; color: red;'>2022 Table with year updated</h1>", unsafe_allow_html=True)
    pre22df['year'] = pd.DatetimeIndex(pre22df['release_date']).year
    pre22df.drop(drop_columns, axis = 1, inplace = True)
    st.write(pre22df.head())

    # Train model
    chosen_features = ['danceability','energy','loudness','speechiness','valence','tempo','artist_pop']
    xtrain = traindf[chosen_features]
    ytrain = traindf['track_pop']
    regr = linear_model.LinearRegression()
    regr.fit(xtrain, ytrain)

    # 2021 Prediction
    xpred = predf[chosen_features]
    ypred = predf['track_pop']
    pred = regr.predict(xpred)
    mse=mean_squared_error(ypred,pred)
    st.markdown("<h1 style='text-align: left; color: red;'>Mean Sqare Error of 2021 </h1>", unsafe_allow_html=True)
    st.write(mse)
    predf['popularity_prediction'] = pred
    st.markdown("<h1 style='text-align: left; color: red;'>Updated table of 2021 </h1>", unsafe_allow_html=True)
    st.write(predf.head())

    # Top 10 most popular song in 2021 prediction
    st.markdown("<h1 style='text-align: left; color: red;'>Top 10 of 2021 </h1>", unsafe_allow_html=True)
    top_10 = sqldf('SELECT artist, song_name, track_pop FROM predf where year = 2021 ORDER BY track_pop DESC limit 10')
    st.write(top_10)

    # Top 10 songs prediction popularity score
    st.markdown("<h1 style='text-align: left; color: red;'> Prediction of 2021 </h1>", unsafe_allow_html=True)
    top_10_predict = sqldf('SELECT artist, song_name, popularity_prediction FROM predf WHERE year = 2021 ORDER BY popularity_prediction DESC')
    st.write(top_10_predict.head(10))

    # 2022 Prediction
    xpred = pre22df[chosen_features]
    ypred = pre22df['track_pop']
    pred = regr.predict(xpred)
    mse=mean_squared_error(ypred,pred)
    st.markdown("<h1 style='text-align: left; color: red;'>Mean Sqare Error of 2022 </h1>", unsafe_allow_html=True)
    st.write(mse)
    pre22df['popularity_prediction'] = pred
    st.markdown("<h1 style='text-align: left; color: red;'>Updated table of 2022 </h1>", unsafe_allow_html=True)
    st.write(pre22df.head())

    # Spotify top 10 of 2022
    st.markdown("<h1 style='text-align: left; color: red;'>Top 10 of 2022 </h1>", unsafe_allow_html=True)
    top_10 = sqldf('SELECT artist, song_name, track_pop FROM pre22df WHERE year = 2022 ORDER BY track_pop DESC')
    st.write(top_10.head(10))

    # My prediction of top 10 of 2022
    st.markdown("<h1 style='text-align: left; color: red;'> Prediction of 2022 </h1>", unsafe_allow_html=True)
    top_10_predict = sqldf('SELECT artist, song_name, popularity_prediction FROM pre22df WHERE year = 2022 ORDER BY popularity_prediction DESC')
    st.write(top_10_predict.head(10))

    st.markdown("<h1 style='text-align: left; color: red;'>2021 and 2022 popularity predictions</h1>", unsafe_allow_html=True)
    st.write("We have built a simple Linear Regression model through scikit-learn, using these following variables: 'danceability','energy','loudness','speechiness','acousticness','liveness','valence','tempo', 'artist_pop'. We decided to remove keys, mode and instrumentalness since they show little correlation to the output popularity we are predicting. We have outputed our own top 10 charts for those two years.")

page_names_to_funcs = {
    "Visualization": main_page,
    "Recommedation System": page2,
    "2021-2022 Prediction": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
