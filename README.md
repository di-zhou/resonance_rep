# resonance_rep
Replication code and data 

There are five folders containing code used in different steps of the project. 

* The [1_scraping](/1_scraping) folder contains code used to scrape data from the forum. They are in the format of Jupyter Notebook.
* The [2_segment](/2_segment) folder contains code used to process the scraped data, including: saving each post as a individual .txt file with consistent naming rules, performing Jieba segmentation to these posts (normal segmentation and POS-tag based segmentation). Please read the comments closely before running these files. They are in the format of either R or Python scripts. 
* The [3_diachronic_w2v](/3_diachronic_w2v) folder contains code (Python script format) used to run:
    - The "start model" (based on text from 2011 to 2017)
    - The w2v model using min_word_freq = 0 and window_size = 20 (based on text from 2016 to 2017)
    - The w2v model using min_word_freq = 10 and window_size = 20 (based on text from 2016 to 2017, used for sensitivity test)
* The [4_sentiment](/4_sentiment) folder contains code used to generate sentiment measure for each post. The file is in Python script format. 
* The [5_analysis](/5_analysis) folder contains code and data used for modeling and plotting, as well as code for sensitivity tests. The code is in R Markdown format. 
