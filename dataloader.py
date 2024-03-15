import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
import ast

class CustomDataset(Dataset):
    def __init__(self,csv_file,transform=None,target_transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        tokens = ast.literal_eval(self.data.iloc[idx]['token'])
        labels = int(self.data.iloc[idx]['label'])
        features = torch.tensor(tokens, dtype=torch.float32)
        label = torch.tensor(labels, dtype=torch.int)

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        return features,labels

def CustomDataloader(csv_file,batch_size,split_size=0.2,shuffle=True):
    dataset = CustomDataset(csv_file)

    print("""
    =============================================================================
    Dataset loaded...
    =============================================================================
    """)


    train_size = int((1-split_size)*len(dataset))
    test_size = len(dataset) - train_size

    train_data,test_data = random_split(dataset,[train_size,test_size])

    val_size = int((split_size)*train_size)
    train_size = train_size - val_size

    #######################################################################################################################################
    print("Train Size: ", train_size)
    print("Test Size: ", test_size)
    print("Validation Size: ", val_size)

    print("""
    =============================================================================
    Splitting Train, Validation and Test Data...
    =============================================================================
    """)
    #######################################################################################################################################

    train_data,val_data = random_split(train_data,[train_size,val_size])
    
    batch_size = batch_size
    shuffle = shuffle

    train_loader = DataLoader(train_data,batch_size,shuffle)
    val_loader = DataLoader(val_data,batch_size,shuffle)
    test_loader = DataLoader(test_data,batch_size,shuffle)


    return train_loader,val_loader,test_loader



# if __name__ == "__main__":
#     csv_file =  "Tokenized_Outputs/Canine_tokenized.csv"
#     
#     train_loader,val_loader,test_loader = CustomDataloader(csv_file)
#
#
#     print("""
#     =============================================================================
#     Iterating Through Dataset...
#     =============================================================================
#     """)
#
#     for batch_idx, (features,labels) in enumerate(train_loader):
#
#         print(f"Batch {batch_idx}:")
#         print("Features",features)
#         print("Labels", labels)
#
#         
#     for batch_idx, (features,labels) in enumerate(test_loader):
#
#         print(f"Batch {batch_idx}:")
#         print("Features",features)
#         print("Labels", labels)
#
#     for batch_idx, (features,labels) in enumerate(val_loader):
#
#         print(f"Batch {batch_idx}:")
#         print("Features",features)
#         print("Labels", labels)


