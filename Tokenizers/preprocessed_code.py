import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# Load the dataset
input_csv = pd.read_csv("../idata/Unbiased_cwe476_Data_tp.csv")
df_code = input_csv.drop(columns=['Label','CWE-476'],axis=1)

# Converting to LowerCase
print('------------------Converting to LowerCase----------------')
df_code['clean_code'] = df_code['code'].str.lower()

# Remove punctuations
print('------------------Removing Punctuations----------------')
def remove_punctuations(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('','',punctuations))

df_code['clean_code'] = df_code['clean_code'].apply(lambda x: remove_punctuations(x))

# Remove Stop WorConvertingds
print('------------------Removing Stop Words----------------')
stopwords = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords])

df_code['clean_code'] = df_code['clean_code'].apply(lambda x: remove_stopwords(x))

# Remove Frequent words
word_count = Counter()
for text in df_code['clean_code']:
    for word in text.split():
        word_count[word] +=1

frequent_words = set(word for (word,wc) in word_count.most_common(10))
def remove_freq_words(text):
    return " ".join([word for word in text.split() if word not in frequent_words])


df_code['clean_code'] = df_code['clean_code'].apply(lambda x: remove_freq_words(x))
print(df_code.head())

# Remove Rare Words

rare_words = set(word for (word,wc) in word_count.most_common()[:-50:-1])
print(rare_words)

def remove_rare_words(text):
    return " ".join([word for word in text.split() if word not in rare_words])


df_code['clean_code'] = df_code['clean_code'].apply(lambda x: remove_rare_words(x))
print(df_code.head())

print('------------------Stemming----------------')

ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])

df_code['stemmed_code'] = df_code['clean_code'].apply(lambda x: stem_words(x))

# print(df_code.head())
print('------------------Lamenting----------------')

lemmatizer=WordNetLemmatizer()
wordnet_map = {'N':wordnet.NOUN, "V": wordnet.VERB, "J":wordnet.ADJ, 'R':wordnet.ADV}

def lemmatize_words(text):
    pos_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word,wordnet_map.get(pos[0],wordnet.NOUN)) for word,pos in pos_text])

df_code['lemmatized_code'] = df_code['clean_code'].apply(lambda x: lemmatize_words(x))

df_code = df_code.drop(columns=['stemmed_code','clean_code','code'],axis=1)
df_code.rename(columns={'lemmatized_code':'code'},inplace=True)
print(df_code.head())

df = pd.DataFrame()
df['code'] = df_code['code']
df['Label'] = input_csv['Label']


df.to_csv('../Tokenized_Outputs/pre_processed_code_min_max_remd.csv', index=False)
df.head()
