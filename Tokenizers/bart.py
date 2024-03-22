import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import BartTokenizer


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

# frequent_words = set(word for (word,wc) in word_count.most_common(3))
# def remove_freq_words(text):
#     return " ".join([word for word in text.split() if word not in frequent_words])
#
#
# df_code['clean_code'] = df_code['clean_code'].apply(lambda x: remove_freq_words(x))
# print(df_code.head())

# Remove Rare Words

# rare_words = set(word for (word,wc) in word_count.most_common()[:-50:-1])
# print(rare_words)
#
# def remove_rare_words(text):
#     return " ".join([word for word in text.split() if word not in frequent_words])
#
#
# df_code['clean_code'] = df_code['clean_code'].apply(lambda x: remove_rare_words(x))
# print(df_code.head())
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

# Check the Tokenization
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence, add_special_tokens=True, padding="max_length", truncation=True, max_length=512)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Sample Sequence:", sequence)
print("Tokens for Sample Sequence: ",tokens)
print("Generated Vectors for Tokens: ",ids)


# Check tokenization for the first code snippet
sample_text = df_code['code'][0]
code_tokens = tokenizer.tokenize(sample_text)
code_ids = tokenizer.convert_tokens_to_ids(code_tokens)

print("Sample code: ",sample_text)
print("Sample Tokens: ",code_tokens)
print("Sample ids: ",code_ids)

# detokenized_string = tokenizer.decode(code_ids)
# print(detokenized_string)

# tokenization

print("---------------------TOKENIZATION STARTED---------------")

tokenized_code = []
for code in df_code['code']:
    code_tokens = tokenizer.tokenize(code)
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    tokenized_code.append(code_ids)

processed_tokenized_code = []

print("---------------------TOKENIZATION FINISHED---------------")
# Define the maximum length threshold
max_length_threshold = 2048

print("---------------------SETTING THE VECTORS TO SAME LENGTH---------------")
for sublist in tokenized_code:
    if len(sublist) < max_length_threshold:
        # Pad the sublist with zeros if its length is less than the threshold
        padded_sublist = sublist + [0] * (max_length_threshold - len(sublist))
        processed_tokenized_code.append(padded_sublist)
    else:
        # Take the first 1024 elements of the sublist if its length exceeds the threshold
        truncated_sublist = sublist[:max_length_threshold]
        processed_tokenized_code.append(truncated_sublist)

print("---------------------CONVERTING IN TO ARRAY---------------")
# Convert the processed nested list to a NumPy array
tokenized_code_array = np.array(processed_tokenized_code)

# Check the shape of the array
print("Shape of the array:", tokenized_code_array.shape)
print("Sample tokenized code: ",tokenized_code_array[18316])


label = input_csv['Label']
# print(label.shape)

print("---------------------CONVERTING INTO DATAFRAME---------------")
df = pd.DataFrame({'token': tokenized_code_array.tolist(), 'label': label})

# Converting to csv and saving it
df.to_csv('../Tokenized_Outputs/Bart_tokenized.csv', index=False)
df.head()
