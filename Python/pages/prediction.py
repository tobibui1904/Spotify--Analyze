import pandas as pd
from pandasql import sqldf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import streamlit as st

st.markdown("<h1 style='text-align: center; color: lightblue;'>2021 and 2022 Prediction</h1>", unsafe_allow_html=True)

# Upload dataset
traindf = pd.read_csv(r"C:\Users\Admin\econ312prj\features.csv")
predf = pd.read_csv(r"C:\Users\Admin\econ312prj\2021.csv")
pre22df = pd.read_csv(r"C:\Users\Admin\econ312prj\2022.csv")
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
st.write(mse)
predf['popularity_prediction'] = pred
st.write(predf.head())

# Top 10 most popular song in 2021
top_10 = sqldf('SELECT artist, song_name, track_pop FROM predf where year = 2021 ORDER BY track_pop DESC limit 10')
st.write(top_10)

# Top 10 songs prediction popularity score
top_10_predict = sqldf('SELECT artist, song_name, popularity_prediction FROM predf WHERE year = 2021 ORDER BY popularity_prediction DESC')
st.write(top_10_predict.head(10))

# 2022 Prediction
xpred = pre22df[chosen_features]
ypred = pre22df['track_pop']
pred = regr.predict(xpred)
mse=mean_squared_error(ypred,pred)
st.write(mse)
pre22df['popularity_prediction'] = pred
st.write(pre22df.head())

# Spotify top 10 of 2022
top_10 = sqldf('SELECT artist, song_name, track_pop FROM pre22df WHERE year = 2022 ORDER BY track_pop DESC')
st.write(top_10.head(10))

# My top 10 of 2022
top_10_predict = sqldf('SELECT artist, song_name, popularity_prediction FROM pre22df WHERE year = 2022 ORDER BY popularity_prediction DESC')
st.write(top_10_predict.head(10))