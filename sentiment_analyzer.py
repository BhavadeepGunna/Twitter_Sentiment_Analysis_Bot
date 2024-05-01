import json
import re
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from tweety.types import Tweet



template="""
you are expert in conducting a sentiment analysis of tweets in the twitter. you know how to classify the tweet given by a user based on the sentiment.
you are given tweets :{tweets}.
Tell me the percentage of positve tweets and negeative tweets for each date. use numbers betweeen 0 and 100 for the percentage.

give the response out in JSON format:
(date:positive_tweets_percentage,negative_tweets_percentage).
Each record of the JSON should give the aggreagate sentiment of the date.Return just the JSON. DO not explain/.

"""
def clean_tweet(text:str)->str:
    text=re.sub(r"http\S+","",text)
    text=re.sub(r"www.\S+","",text)
    return re.sub(r"\s+"," ",text)

def create_dataframe_from_tweets(tweets: List[Tweet])->pd.DataFrame:
    rows=[]
    for tweet in tweets:
        clean_text=clean_tweet(tweet.text)
        if len(clean_text)==0:
            continue
        rows.append(
            {
                "id":tweet.id,
                "text":clean_text,
                "author":tweet.author.username,
                "date":str(tweet.date.date()),
                "created_at":tweet.date,
                "views":tweet.views,

            }
        )
    
    df=pd.DataFrame(
        rows,columns=['id','text','author','date','views','created_at']
    )
    df.set_index('id',inplace=True)

    if df.empty:
        return df
    df=df[df.created_at.dt.date>datetime.now().date()-pd.to_timedelta('7day')]
    return df.sort_values(by="created_at",ascending=False)


def create_tweet_list(tweets,date)->list:
    if tweets.empty:
        return""
    
    res=[]

    text=""
    text+=f"{date}:"
    for t in tweets.itertuples():
        if(len(text+f"\n{t.views}-{t.text}")<16000):
            text+=f"\n{t.views}-{t.text}"
        else:
            res.append(text)
            text=f"{date}:"
            text+=f"\n{t.views}-{t.text}"
    res.append(text)
    return res


def create_response(input):
    chat_gpt=ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")
    prompt=PromptTemplate(
        input_variables=['tweets'],
        template=template
    )
    sentiment_chain=LLMChain(llm=chat_gpt,prompt=prompt)
    response=sentiment_chain(
            {
            "tweets":input
            }
        )
    
    return json.loads(response['text'])


def analyze_sentiment(tweets,date)->Dict[str,int]:
    
    chunks_texts=create_tweet_list(tweets,date)
    result={}
    j=1
    for input in chunks_texts:
        result[j]=create_response(input)
        j+=1

    positive=0
    negative=0
    for i in range(1,j):
        curr=result[i]
        positive+=result[i][date]['positive_tweets_percentage']
        negative+=curr[date]['negative_tweets_percentage']
    if j!=1:
        positive=positive/(j-1)
        negative=negative/(j-1)

    #print(positive,negative)


    return {date:{'positive_tweets_percentage':positive,
                  'negative_tweets_percentage':negative}}
