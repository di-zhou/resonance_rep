# Clean and tokenize (segment) answer post

# The input are text files obtained from "time_slice_posts.R"
# Use this after running the script "time_slice_posts.R"
# Remember to change the file path name according to your file structure

# Author: Di Zhou
# Last run: Nov. 2020 


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

import jieba
import jieba.analyse as analyse
import jieba.posseg as pseg
## user-defined new word list
jieba.load_userdict('data/jieba_user_dict.txt')

import csv
import re

def tokenization(content):
    stop_flags = {'x'}
    stop_words = {}
    words = pseg.cut(content)
    return [word for word, flag in words if flag not in stop_flags and word not in stop_words]

def seg_txt_from_path(input_path, output_path):
    '''
    clean & tokenize text in the input file, save text to output file
    input: input txt file path
    output: output txt file path
    '''
    with open(input_path,'r') as f:
        for line in f:
            new_line = re.sub(r"http\S+|[0-9.]+", "", line) #remove URL & numbers & dots
            seg = tokenization(new_line) # tokenize (above function)
            output = ' '.join(seg)
            output = output + ""
            with open(output_path,'a+')as s:
                s.write(output)

# Construct a list of input and output file path
file_path_ls = [f for f in os.listdir('./data/chrono_post_data/') if f.endswith(".txt")]
# Iterate through files
for path in file_path_ls:
    input_path = './data/chrono_post_data/' + path
    output_path = './data/chrono_post_seg/' + path
    seg_txt_from_path(input_path, output_path)






