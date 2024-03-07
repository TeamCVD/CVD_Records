import torch
import torch.nn as nn
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


# Same as linear regression! 
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x
    
input_dim = 2048
output_dim = 1

model = LogisticRegressionModel(input_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss() 

learning_rate = 0.0008

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0

    for inputs, targets in zip(x_train, y_train):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(0))
        # print("current Loss: ",loss)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        # print("running loss: ",running_loss)

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