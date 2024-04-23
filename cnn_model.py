import torch.nn as nn


class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 7, 1, 3)
        self.conv2 = nn.Conv1d(8, 32, 7, 1, 3)
        self.max_pool1 = nn.MaxPool1d(2, 2)
        self.batch_norm1 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 128, 5, 1, 2)
        self.conv4 = nn.Conv1d(128, 256, 5, 1, 2)
        self.conv5 = nn.Conv1d(256, 256, 5, 1, 2)
        self.max_pool2 = nn.MaxPool1d(2, 2)
        self.batch_norm2 = nn.BatchNorm1d(256)

        self.conv6 = nn.Conv1d(256, 128, 5, 1, 2)
        self.conv7 = nn.Conv1d(128, 64, 3, 1, 1)
        self.max_pool3 = nn.MaxPool1d(2, 2)
        self.batch_norm3 = nn.BatchNorm1d(64)

        self.conv8 = nn.Conv1d(64, 16, 3, 1, 1)
        self.conv9 = nn.Conv1d(16, 8, 3, 1, 1)
        self.conv10 = nn.Conv1d(8, 4, 3, 1, 1)
        self.max_pool4 = nn.MaxPool1d(2, 2)
        self.batch_norm4 = nn.BatchNorm1d(4)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,128)
        self.fc5 = nn.Linear(128,16)
        self.fc6 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.max_pool1(x)
        x = self.batch_norm1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool2(x)
        x = self.batch_norm2(x)

        x = self.conv6(x)
        x = self.conv7(x)
        # x = self.max_pool3(x)
        x = self.batch_norm3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool4(x)
        x = self.batch_norm4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        # x = self.leakyRelu(x)
        # x = self.fc5(x)
        # x = self.fc6(x)
        x = self.sigmoid(x)

        return x
