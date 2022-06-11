# Get novelty measure using diachronic word embedding
# w2v with min_count = 0, window = 20

# Remember to change the file path name according to your file structure
# This script takes relatively long time to run. Cloud computing suggested.

# Author: Di Zhou
# Last run: Nov. 2020

# -*- coding: utf-8 -*-
import numpy as np
import re
import os
import pickle
import math

import gensim.models
from gensim import utils
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

from sklearn.utils import resample

'''
This uses w2v with min_count = 0, window = 20
'''


def get_post_novelty(post_token_list, w2v_model):
    '''
    Obtain novelty measure from a post's token list
    input: a post's token list, a w2v model
    output: novelty measure of the post
    '''
    post_token_unique = []
    [post_token_unique.append(v) for v in post_token_list if v not in post_token_unique]
    # get post vector list
    vector_ls = []
    for i in post_token_unique:
        if i in w2v_model.wv.vocab:
            vector_ls.append(w2v_model[i])
            
    if len(vector_ls) == 0:
        novelty = 'NA'
        
    if len(vector_ls) > 0:
        # Calculate novelty measure based on post vector list
        dimension=vector_ls[0].shape[0]
        dimension_sum=0
        for d in range(dimension):
            vector_multi=1
            for i in range(len(vector_ls)):
                vector_multi=vector_multi*vector_ls[i][d]
            dimension_sum+=vector_multi           
        novelty=-np.log(abs(dimension_sum))

    return novelty
    
# Obtain time slice from file name
file_path_ls_t = [f for f in os.listdir("./data/chrono_post_seg/") if f.endswith(".txt")] 
time_slice = []
for i in file_path_ls_t:
    i = re.sub(r'(?<=[wk\d|wk\d{2}])\_\d+\.txt', '', i) # keep only year_week
    if i not in time_slice:                             # drop duplicates
        time_slice.append(i)
time_slice = sorted(time_slice)


# Iteratively measure novelty with bootstraping
# ls_of_array = []
nov_stats = []
n_iter = 150

for idx in range(1,len(time_slice)):
    
    current_t = time_slice[idx]
    previous_t = time_slice[idx - 1]
    # define the previous model's file name
    if idx <= 1:
        previous_model_fp = './model_min0_win20/model_start.bin'
    if idx > 1:
        previous_model_fp = './model_min0_win20/model_' + time_slice[idx-2] + '.bin'
    
    
    # corpus of posts in previous tims slice (for training model)
    previous_t_post_fn = sorted([f for f in os.listdir("./data/chrono_post_seg/") if previous_t in f]) 
    previous_t_corpus = [] # 1-level: post, 2-level: tokens
    for fp in previous_t_post_fn:
        post_fp = open('./data/chrono_post_seg/' + fp)
        post = post_fp.read().replace('\n', '')
        previous_t_corpus.append(post.split(' ')) 
    
    
    # a nested list of post and tokens of current time slice (for novelty measure), only NAV POS tags are kept
    current_t_post_fn = [f for f in os.listdir("./data/chrono_post_seg_NAVonly/") if current_t in f]
    current_t_post_fn.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'(?<=\d{4}\_wk\d{2}\_)\d+(?=\.txt)', var)])    
    n_post_in_current_t = len(current_t_post_fn)   # get the number of post in the time slice (for array shape)
    current_post_token_list = []                   # 1-level: post, 2-level: tokens
    for fp in current_t_post_fn:
        post_fp = open('./data/chrono_post_seg_NAVonly/' + fp)
        post = post_fp.read().replace('\n', '')
        current_post_token_list.append(post.split(' ')) 
        
    array = np.empty([n_iter, n_post_in_current_t])  # empty array to save novelty stats
    
    for k in range(n_iter):
        sentence_samples = resample(previous_t_corpus) # resample posts from last week's corpus
        model = Word2Vec.load(previous_model_fp)
        # print('current time slice', current_t, 'previous_model', previous_model_fp, 'train on', previous_t)
        model.train(sentence_samples, total_examples = len(sentence_samples), epochs = n_iter)
        
        for i in range(len(current_post_token_list)): # for each post
            novel = get_post_novelty(current_post_token_list[i], model)
            if novel is not 'NA':
                array[k, i] = novel
            if novel is 'NA':
                array[k, i] = np.nan
            
        run = k+1

    # save the last iteration as the starting model for next time slice   
    new_model_fp = './model_min0_win20/model_' + previous_t + '.bin'
    model.save(new_model_fp)
    
    for j in range(len(current_post_token_list)):
        post_idx = current_t + "_" + str(j+1)
        bootmean = array[:,j].mean()
        bootstd = array[:,j].std()
        ci_upper = np.percentile(array[:,j], 95, axis = 0) # 90% CI upper
        ci_lower = np.percentile(array[:,j], 5, axis = 0)  # 90% CI lower
        nov_stats.append([post_idx, bootmean, bootstd, ci_upper, ci_lower]) # append post novelty to list
    
    # ls_of_array.append(array)
    
# save results
with open('./data/nov_stats_min0_win20.pkl', 'wb') as f:
    pickle.dump(nov_stats, f)
# with open('./data/list_of_boot_array.pkl', 'wb') as f:
#    pickle.dump(ls_of_array, f)



