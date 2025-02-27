# 导入所需的包
library(tm)
library(wordcloud)
library(wordcloud2)
# 读取文本文件
text <- readLines("F:\\共享文件夹\\统计建模\\Code\\R\\调用文件\\processed_text.txt", warn = FALSE)

# 将文本数据转换成文档对象
corpus <- Corpus(VectorSource(text))

# 文本预处理
corpus <- tm_map(corpus, content_transformer(tolower)) # 转换为小写
corpus <- tm_map(corpus, removePunctuation) # 去除标点符号
corpus <- tm_map(corpus, removeNumbers) # 去除数字
corpus <- tm_map(corpus, removeWords, c(stopwords("english"), "的","和","为")) # 去除停用词和特定单词

# 创建词频矩阵
dtm <- DocumentTermMatrix(corpus)

# 计算词频
word_freq <- colSums(as.matrix(dtm))

# 创建数据框
freq <- data.frame(word = names(word_freq), freq = word_freq)

# 过滤词频低于10的词
freq <- freq[freq$freq >= 5, ]

# 创建一个空的数据框
freq_df <- data.frame(word = character(0), freq = numeric(0))

# 将词频矩阵转换为数据框并添加到freq_df
freq_df <- rbind(freq_df, freq)

# 将 "碳" 和 "排放" 的频率设置为较高
freq_df[freq_df$word == "碳", "freq"] <- 300
freq_df[freq_df$word == "排放", "freq"] <- 200

# 生成词云图
pdf("词云图.pdf")
wordcloud2(data = freq_df, shape = 'circle')