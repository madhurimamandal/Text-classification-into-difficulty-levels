# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Importing the dataset
Dataframe_Final = pd.read_csv('Dataset(final).csv')

#Splitting into features and classes
X = Dataframe_Final.drop('File Name',axis=1).drop('Class',axis=1)
y = Dataframe_Final['Class']

#Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 0, shuffle = False)

# Feature Scaling
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train)
X_test2 = sc.transform(X_test)

#Finding the best value of n_estimators
classacc = []
m = []
for i in range(5,101):
    rfclassifier = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 0)
    rfclassifier.fit(X_train2, y_train)
    y_pred = rfclassifier.predict(X_test2)
    cm = confusion_matrix(y_test,y_pred)
    classacc.append(accuracy_score(y_test, y_pred))
    accuracies = cross_val_score(estimator = rfclassifier, X = X_train2, y = y_train, cv = 10)
    m.append(accuracies.mean())
plt.plot(range(5,101), classacc, label='classification accuracy')
plt.plot(range(5,101), m, label='kfold accuracy')
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.title("Accuracy variance with number of trees")
plt.legend()
plt.show()