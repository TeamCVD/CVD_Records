import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier,self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fce1 = nn.Linear(4096, 2048)
        self.fce2 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fcn = nn.Linear(128,128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8,4)
        self.fc9 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fce1(x))
        x = self.dropout(x)
        x = torch.relu(self.fce2(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fcn(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.sigmoid(self.fc8(x))
        x = self.sigmoid(self.fc9(x))
        return x
 
