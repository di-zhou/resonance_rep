# Save posts as individual .txt files with year and week number in file name

# Remember to change the file path name according to your file structure
# The input is a dataframe with two variables: answer date, answer text
# The output are .txt files for word-embedding models
# Sample .txt file name: /2016_wk49_452.txt

# For "start model", all texts by 2017/12/31 is kept
# For "chrono model", only texts from 2016/01/01 to 2017/12/31 are used
# You should make such changes to the code below to create texts for both start and chrono models

# Author: Di Zhou
# Last run: Nov. 2020 

library(tidyverse)
library(lubridate)
library(data.table)

# Load data
all_answer_text_only <- read.csv("data/all_answer_text_only.csv", 
                                 fileEncoding = 'UTF-8', 
                                 stringsAsFactors = FALSE) 

# A function that convert 1 digit week to two
two_digit_week <- function(date){
  x = lubridate::week(date)
  if (nchar(x) == 1){
    y = paste0(0, x, collapse = "")
  }
  else{
    y = x
  }
  return(y)
}

# Tag year-week to raw data (keep posts before 2017/12/31): 
all_answer_text_only_week <- all_answer_text_only %>%
  # For "start model", all text by 2017/12/31 is kept
  # For "chrono model", only text from 2016/01/01 to 2017/12/31 are used
  # You should make such changes to this part of the code (and file path names) 
  # to create texts for both start and chrono models
  filter(as.Date(answer_date) <= '2017/12/31' & as.Date(answer_date) >= '2016/01/01') %>%
  rowwise() %>%
  mutate(year_wk = paste0(year(answer_date), "_wk", two_digit_week(answer_date), collapse = ""))

# Save each post to file path "data/chrono_post_data"

# A seq of year week
year_wk_ls <- unique(all_answer_text_only_week$year_wk) %>% sort() 

# Loop along the seq
for (i in 1:length(year_wk_ls)){
  year_week <- year_wk_ls[i]
  
  post_of_yrwk <- all_answer_text_only_week %>%
    filter(year_wk == year_week) %>%
    arrange(answer_date)
  
  for (j in 1:nrow(post_of_yrwk)){
    post <- as.character(post_of_yrwk$answer_content[j])
    post_id <- post_of_yrwk$answer_id[j]
    post_filename <- paste(year_week, "_", j, ".txt", sep = "")
    
    # a file of year_wk tag mapping to original answer id 
    cat(c(post_id, post_filename), file = "data/chrono_post_key.csv", append = TRUE, sep = ",", eol = "\n")
    
    fileConn <- file(paste("data/chrono_post_data/", year_week, "_", j, ".txt", sep = ""))
    writeLines(post, fileConn)
    close(fileConn)
    
  }
  
}


