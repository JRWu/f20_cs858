#!/usr/bin/R
"""
install.packages("wordcloud")
install.packages("RColorBrewer")
install.packages("wordcloud2")
install.packages("tm")

"""
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(dplyr)



data_dir <- "/home/jia/Documents/Code/UW/f20_cs858/data"
first_file <- "1_sleep_cycle_sleep_tracker_sleep_cycle_ab.txt"


policies <- list.files(data_dir, pattern="*.txt")

for (pol in policies){
	fin <- paste(data_dir, pol, sep="/")
	pol_data <- scan(fin, character(), quote = "")
	pol_corpus <- Corpus(VectorSource(pol_data))

	pol_corpus <- pol_corpus %>%
		tm_map(removeNumbers) %>%
		tm_map(removePunctuation) %>%
		tm_map(stripWhitespace)
	pol_corpus <- tm_map(pol_corpus, content_transformer(tolower))
	pol_corpus <- tm_map(pol_corpus, removeWords, stopwords("english"))


	dtm <- TermDocumentMatrix(pol_corpus) 
	matrix <- as.matrix(dtm) 
	words <- sort(rowSums(matrix),decreasing=TRUE) 
	df <- data.frame(word = names(words),freq=words)
	
	out_png <- (gsub(".txt", ".png", pol))

	set.seed(1234) # for reproducibility 

	png(filename=paste(data_dir,out_png,sep="/"))
	wordcloud(words = df$word, freq = df$freq, min.freq = 1,           max.words=200, random.order=FALSE, rot.per=0.35,            colors=brewer.pal(8, "Dark2"))
	dev.off()
}






fin <- paste(data_dir, first_file, sep="/")



first_data <- scan(fin, character(), quote = "")


#first_corpus <- strsplit(first_data, "\\s+")[[1]]

first_corpus <- Corpus(VectorSource(first_data))



first_corpus <- first_corpus %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
first_corpus <- tm_map(first_corpus, content_transformer(tolower))
first_corpus <- tm_map(first_corpus, removeWords, stopwords("english"))


dtm <- TermDocumentMatrix(first_corpus) 
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df <- data.frame(word = names(words),freq=words)

set.seed(1234) # for reproducibility 
wordcloud(words = df$word, freq = df$freq, min.freq = 1,           max.words=200, random.order=FALSE, rot.per=0.35,            colors=brewer.pal(8, "Dark2"))
