# Train "start model" 

# This uses segmented text obtained using the script "seg_post_startmodel.py"
# Remember to change the file path name according to your file structure

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

Original setup: min_count = 0, window = 20

Sensitivity test: min_count = 10, window = 20

'''


# Segmented post filenames
file_path_ls = [f for f in os.listdir("./data/chrono_post_startmodel_seg/") if f.endswith(".txt")]
# A nested list: level 1 = post, level 2 = token
agg_corpus = []
for i in file_path_ls:
    path = './data/chrono_post_startmodel_seg/' + i
    fp = open(path)
    x = fp.read().replace('\n', '')
    agg_corpus.append(x.split(' '))

# Train w2v model:
model_start_min0_win20 = Word2Vec(agg_corpus, size = 100, min_count = 0, iter = 200,
                     sg = 1, hs = 0, negative = 5, window = 20, workers = 20)
# Save model
model_start_min0_win20.save('./model_min0_win20/model_start.bin')

# Train:
model_start_min10_win10 = Word2Vec(agg_corpus, size = 100, min_count = 10, iter = 200,
                     sg = 1, hs = 0, negative = 5, window = 20, workers = 20)
# Save model
model_start_min10_win10.save('./model_min10_win20/model_start.bin')






