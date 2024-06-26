import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier,self).__init__()
        self.fc1 = nn.Linear(1024,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fce1 = nn.Linear(2048, 1024)
        self.fce2 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
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
        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.relu(self.fce1(x))
        x = self.dropout(x)
        x = self.fce2(x)
        x = torch.relu(self.fc2(x))
        # x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.relu(self.fcn(x))
        x = self.fc4(x)
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        x = self.sigmoid(self.fc9(x))
        print(x.shape)
        return x
 
