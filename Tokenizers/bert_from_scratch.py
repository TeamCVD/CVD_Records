from tokenizers import Tokenizer
from tokenizers.models import WordPiece
import pandas as pd 

bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

from tokenizers import normalizers
from tokenizers.normalizers import NFD,Lowercase,StripAccents

bert_tokenizer.normalizer = normalizers.Sequence([NFD(),Lowercase(),StripAccents()])


from tokenizers.pre_tokenizers import Whitespace

bert_tokenizer.pre_tokenizer = Whitespace()

from tokenizers.processors import TemplateProcessing

bert_tokenizer.post_processors = TemplateProcessing (
    single = '[CLS] $A [SEP]',
    pair = '[CLS] $A [SEP] $B:1 [SEP]:1',
    special_tokens = [
        ("[CLS]",1),
        ("[SEP]",2),
    ],
)

from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=30522,special_tokens=['[UNK]','[CLS]','[SEP]','[PAD]','[MASK]'])

df = pd.read_csv('../Tokenized_Outputs/Pre_processed_code.csv')
print(df.head())

txt = ''
for code in df['code']:
    txt += code


def string_to_raw_file(string_data, output_file_path):
    with open(output_file_path, 'w') as file:
        file.write(string_data)

#

# Example output file path
output_file_path = "./data/code.raw"

# Call the function to write the string to the .raw file
string_to_raw_file(txt, output_file_path)


file = './data/code.raw'
bert_tokenizer.train(file,trainer)
bert_tokenizer.save("data/bert.json")


