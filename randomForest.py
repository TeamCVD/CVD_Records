import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ast

#Build the model
clf = RandomForestClassifier(max_depth=100, random_state=42)
# clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)
