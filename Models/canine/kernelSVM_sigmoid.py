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


print("---------------------Training Started--------------")

from sklearn.svm import SVC
classifier = SVC(kernel ='sigmoid')
 # training set in x, y axis
classifier.fit(X_train, Y_train)

print("---------------------Training Completed--------------")

print("---------------------Predicting Test Results--------------")

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

print("---------------------Predicting Test Results Completed--------------")

print("---------------------Calculating metrics--------------")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# Accuracy, Precision, Recall, F1 Score
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

