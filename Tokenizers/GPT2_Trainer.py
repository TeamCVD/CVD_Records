import pandas as pd
from datasets import Dataset

# Load CSV file into pandas DataFrame
df = pd.read_csv('../Tokenized_Outputs/pre_processed_cwe476.csv')

# Convert pandas DataFrame to dataset object
dataset = Dataset.from_pandas(df)

# Optional: Inspect the first few examples
print(dataset['code'][:5])  # Replace 'column_name' with the actual column name(s) you want to inspect


batch_size = 2048
all_texts = [dataset[i : i + batch_size]["code"] for i in range(0, len(dataset), batch_size)]


def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["code"]


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=25000)

out = new_tokenizer(dataset[:5]["code"])

new_tokenizer.save_pretrained("my-new-tokenizer")

