import os
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from tweety import Twitter

from sentiment_analyzer import analyze_sentiment, create_dataframe_from_tweets






st.set_page_config(
    layout="wide",
    page_title="Twitter Sentiment Analysis Bot",
    #page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f4c8.png",
)

def on_add_tweets(username,password):
    twitter_client = Twitter('session')
    twitter_client.sign_in(username,password)
    topic=st.session_state.topic
    all_tweets= twitter_client.search(keyword=topic,pages=40)
    if len(all_tweets)==0:
        return
    st.session_state.tweets.extend(all_tweets)

def create_sentiment_dataframe(sentiment_data) -> pd.DataFrame:
    dates=[]
    positive=[]
    negeative=[]
    for date,data1 in sentiment_data.items():
        for date,data in data1.items():
            dates.append(date)
            positive.append(data['positive_tweets_percentage'])
            negeative.append(data['negative_tweets_percentage'])

    chart_data={}
    chart_data['dates']=dates
    chart_data['positive']=positive
    chart_data['negative']=negeative
    sentiment_df=pd.DataFrame(chart_data)
    sentiment_df.set_index("dates", inplace=True)
    return sentiment_df





st.markdown(
    "<h1 style='text-align: center'>Twitter Sentiment Analysis Bot</h1>",
    unsafe_allow_html=True,
)


if not "tweets" in st.session_state:
    st.session_state.tweets = []
    st.session_state.api_key = ""
    st.session_state.sentiment={}
    st.session_state.topic=""

os.environ["OPENAI_API_KEY"] = st.session_state.api_key

col1, col2 = st.columns(2)

with st.sidebar:
    username=st.text_input('enter your username of twitter')
    password=st.text_input('enter your password of your twitter account')
with col1:
    st.text_input(
        "OpenAI API Key",
        type="password",
        key="api_key",
        placeholder="sk-...4242",
        help="Get your API key: https://platform.openai.com/account/api-keys",
    )

    with st.form(key="topic_form", clear_on_submit=True):
        st.subheader("add topic that you want to know about", anchor=False)
        st.text_input(
            "topic", value="", key="topic", placeholder="give the topic"
        )
        submit = st.form_submit_button(label="Add Tweets", on_click=on_add_tweets(username,password))
    if st.session_state.topic:
        st.subheader("Topic", anchor=False)
        name=st.session_state.topic
        st.markdown(f"{name}")
    st.subheader("Tweets", anchor=False)
    tweets_df=create_dataframe_from_tweets(st.session_state.tweets)
    st.dataframe(
        create_dataframe_from_tweets(st.session_state.tweets), use_container_width=True
    )

create=st.button('create')

with col2:
    tweets_df.drop_duplicates(inplace=True)
    for date,data in tweets_df.groupby('date'):
        st.session_state.sentiment[date]=analyze_sentiment(data,date)
    sentiment_df=create_sentiment_dataframe(st.session_state.sentiment)

    if not sentiment_df.empty:
        fig = px.bar(
            sentiment_df,
            x=sentiment_df.index,
            y=sentiment_df.columns,
            labels={"date": "Date", "value": "Sentiment"},
            color_discrete_map={"positive": "green", "negative": "red"}
        )
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        st.dataframe(sentiment_df, use_container_width=True)

        st.write('total number of tweets taken:{}'.format(tweets_df.shape[0]))