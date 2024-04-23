import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers import CanineTokenizer
from transformers import AutoTokenizer
input_csv = pd.read_csv("../Tokenized_Outputs/Pre_processed_code.csv")

# tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
tokenizer = AutoTokenizer.from_pretrained("my-new-tokenizer")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Sample Sequence:", sequence)
print("Tokens for Sample Sequence: ",tokens)
print("Generated Vectors for Tokens: ",ids)

input_csv.head()

sample_text = input_csv['code'][0]
code_tokens = tokenizer.tokenize(sample_text)
code_ids = tokenizer.convert_tokens_to_ids(code_tokens)

print("Sample code: ",sample_text)
print("Sample Tokens: ",code_tokens)
print("Sample ids: ",code_ids)

tokenized_code = []
max_length_threshold = 1024
for code in input_csv['code']:
    code_tokens = tokenizer.tokenize(code)
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    tokenized_code.append(code_ids)

processed_tokenized_code = []
for sublist in tokenized_code:
    if len(sublist) < max_length_threshold:
        padded_sublist = sublist + [0] * (max_length_threshold - len(sublist))
        processed_tokenized_code.append(padded_sublist)
    else:
        truncated_sublist = sublist[:max_length_threshold]
        processed_tokenized_code.append(truncated_sublist)

tokenized_code_array = np.array(processed_tokenized_code)


pre_processed_tokenized_code = torch.tensor(tokenized_code_array).float()

# Define the transformer encoder
encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

encoded_code = []
for code in pre_processed_tokenized_code:
    code = code.unsqueeze(0)
    out = transformer_encoder(code)
    out = out.squeeze(0)
    encoded_code.append(out)

# encoded_code_array
print("Sample Encoded Code: ",encoded_code[12345])
print("Shape of Encoded Code: ",encoded_code[12345].shape)

# Save the encoded code to a NumPy array
encoded_code_array = torch.stack(encoded_code).detach().numpy()
print("Shape of Encoded Code Array: ",encoded_code_array.shape)

label = input_csv['Label']
print(label.shape)

df = pd.DataFrame({'token': encoded_code_array.tolist(), 'label': label})

# Save the encoded code and labels to a CSV file
df.to_csv("../Tokenized_Outputs/Encoded_code.csv", index=False)
df.head()


