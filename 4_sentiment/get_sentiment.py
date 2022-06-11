#!/usr/bin/env python
# coding: utf-8

# # Get sentiment score for each post
# 
# * **Input**: A dataframe (df) with all answer posts and answer post IDs
# * **Output**: A dataframe with two variables: answer post ID and sentiment score
# * **Package**: https://github.com/bung87/bixin


import pandas as pd
import csv
from bixin import predict


def get_answer_list(file_path):
    df = pd.read_csv(file_path, header=0, low_memory=False) 
    df = df[df['answer_content'].notnull()] #drop NaN
    answer_list = df['answer_content'].values.tolist()
    return answer_list

def get_answerID_list(file_path):
    df = pd.read_csv(file_path, header=0, low_memory=False) 
    df = df[df['answer_content'].notnull()] #drop NaN
    answerID_list = df['answer_id'].values.tolist()
    return answerID_list

def get_sentiment_list(answer_list):
    score_list = []
    for answer in answer_list:
        score = predict(answer)
        score_list.append(score)
    return score_list

answer_list = get_answer_list('data/all_answer_text_only.csv')
answerID_list = get_answerID_list('data/all_answer_text_only.csv')
score_list = get_sentiment_list(answer_list)


# Export as csv
data_tuples = list(zip(answerID_list, score_list))
sentiment_df = pd.DataFrame(data_tuples, columns=['answer_id','sentiment'])
sentiment_df.to_csv('temp_data/sentiment_df.csv', index=True, sep=',')

