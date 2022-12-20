1. Jupiter Notebook:

Run the try.ipynb to obtain the full dataset. This is a different script than api.ipynb since I use spotipy to crawl the data rather than using API. I have included the three playlist url list on it. The first one is the full data from 2010-2020, the second one is 2021 and the third one is the 2022 playlist. 

I have included the json dataset in case you do not need to run the scraping script (the 1000 dataset takes ~~ 20 minutes to execute)

Then run the prj.ipynb. This will have all the charts we drew. Finally run the prediction.ipynb which I used a naive Linear Regression to output my predictions for top songs in 2021 and 2022. 

2. Python:

Note: python codes use the .csv dataset in the Files folder. Streamlit optimizes csv rather than json so we have to convert to avoid unwanted bugs.

We have already deployed the website hosted by Heroku server. Additionally, we designed the web with Streamlit API by Snowflake. Therefore, you don't need to run the Python code on your chosen IDE becaused it's specifically coded for Streamlit API  Here is a link to it:

https://tobibui-app.herokuapp.com/?fbclid=IwAR1UaE0o3WMKtGex6rvo0C9ibTpFMN9d-qdWWvq57W0CwawHErAors3ILeg
