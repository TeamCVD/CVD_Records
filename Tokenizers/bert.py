import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Load the dataset
input_csv = pd.read_csv("../Tokenized_Outputs/pre_processed_code_min_max_remd.csv")

# Check the Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence, add_special_tokens=True, padding="max_length", truncation=True, max_length=512)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Sample Sequence:", sequence)
print("Tokens for Sample Sequence: ",tokens)
print("Generated Vectors for Tokens: ",ids)

# Check the first few rows of the dataset
input_csv.head()

# Check tokenization for the first code snippet
sample_text = input_csv['code'][0]
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
for code in input_csv['code']:
    code_tokens = tokenizer.tokenize(code)
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    tokenized_code.append(code_ids)

processed_tokenized_code = []

print("---------------------TOKENIZATION FINISHED---------------")
# Define the maximum length threshold
max_length_threshold = 1024

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
df.to_csv('../Tokenized_Outputs/Bert_tokenized.csv', index=False)
df.head()
