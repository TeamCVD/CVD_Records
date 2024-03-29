import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("---------------------READING CSV--------------")

# Load the dataset
df = pd.read_csv("../../Tokenized_Outputs/Canine_tokenized.csv")
df.head()

print("---------------------READING CSV COMPLETED--------------")

df['token'] = np.array(df['token'].apply(eval))
df["label"] = df["label"].astype(int)

print("---------------------SPLITTING TRAIN, TEST DATA--------------")
X_train, X_test, Y_train, Y_test = train_test_split(np.array(df['token'].tolist()), np.array(df['label']), test_size=0.2, random_state=42)


print("Shape of X-train: ",X_train.shape)
print("Shape of X-test: ",X_test.shape)
print("Shape of Y-train: ",Y_train.shape)
print("Shape of Y-test: ",Y_test.shape)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on GPU")
else:
    device = torch.device("cpu")
    print("Training on CPU")

x_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)

x_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(Y_test, dtype=torch.float32, device=device)


print("Shape of X-train tensor: ",x_train.shape)
print("Shape of Y-train tensor: ",y_train.shape)
print("Shape of X-test tensor: ",x_test.shape)
print("Shape of Y-test tensor: ",y_test.shape)

# Model Architecture
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fce1 = nn.Linear(4096, 2048)
        self.fce2 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fce1(x))
        x = self.dropout(x)
        x = torch.relu(self.fce2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.sigmoid(self.fc8(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryClassifier().to(device)
print(model)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

from torchinfo import summary
summary(BinaryClassifier()) 


print("---------------------TRAINING STARTED--------------")

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    for inputs, targets in zip(x_train, y_train):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(0))
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(x_train)}")

print("---------------------FINISHED TRAINING--------------")

print("---------------------MODEL PREDICTION STARTED--------------")
model.eval()
predicted = model(x_test)

print("Predicted classes: ",predicted)


# Convert the predicted values to binary
predicted = (predicted > 0.5).float()

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
precision = precision_score(y_test.cpu(), predicted.cpu())
recall = recall_score(y_test.cpu(), predicted.cpu())
f1 = f1_score(y_test.cpu(), predicted.cpu())

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")