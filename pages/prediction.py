import pandas as pd
from pandasql import sqldf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import streamlit as st

traindf = pd.read_csv(r"C:\Users\Admin\econ312prj\features.csv")
predf = pd.read_csv(r"C:\Users\Admin\econ312prj\2021.csv")
drop_columns = ['type', 'id', 'uri', 'track_href', 'analysis_url','release_date']

# Update traindf with release_date to year and track_pop to more precise
traindf['year'] = pd.DatetimeIndex(traindf['release_date']).year
traindf.drop(drop_columns, axis = 1, inplace = True)
dfmean = traindf['track_pop'].mean()
traindf.loc[traindf['track_pop'] <25, 'track_pop'] = (dfmean + traindf['track_pop'] + traindf['artist_pop'])/2

# Update predf with release_date to year
predf['year'] = pd.DatetimeIndex(predf['release_date']).year
predf.drop(drop_columns, axis = 1, inplace = True)

# Train model
chosen_features = ['danceability','energy','loudness','speechiness','valence','tempo','artist_pop']
xtrain = traindf[chosen_features]
ytrain = traindf['track_pop']
regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
st.write(regr)

# Train model
xpred = predf[chosen_features]
ypred = predf['track_pop']
pred = regr.predict(xpred)
mse=mean_squared_error(ypred,pred)
predf['popularity_prediction'] = pred
st.write(mse)

# Top 10 most popular song
top_10 = sqldf('SELECT song_name, track_pop FROM predf ORDER BY track_pop DESC')

# Top 10 songs prediction popularity score
top_10_predict = sqldf('SELECT song_name, popularity_prediction FROM predf ORDER BY popularity_prediction DESC')
top_10_predict.head(10)
st.write(top_10_predict)