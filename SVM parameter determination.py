#Packages 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
 
#Importing dataset
Dataframe_Final = pd.read_csv('Dataset(final).csv')

#Splitting into features and classes
X = Dataframe_Final.drop('File Name',axis=1).drop('Class',axis=1)
y = Dataframe_Final['Class']
 
#Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 0, shuffle = False)
 
#Scaling features
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train)
X_test2 = sc.transform(X_test)
 
parameters = {'C': [0.1,0.2,0.3,0.4,0.5], 'gamma': [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]}

#Finding the best value of C
classacc = []
m = []
for i in parameters['C']:
    svclassifier = SVC(kernel='rbf', C=i, random_state=0, gamma = 0.5) 
    svclassifier.fit(X_train2, y_train)
    y_pred = svclassifier.predict(X_test2)
    cm = confusion_matrix(y_test,y_pred)
    classacc.append(accuracy_score(y_test, y_pred))
    accuracies = cross_val_score(estimator = svclassifier, X = X_train2, y = y_train, cv = 10)
    m.append(accuracies.mean())
plt.plot(parameters['C'], classacc, label='classification accuracy')
plt.plot(parameters['C'], m, label='kfold accuracy')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Accuracy variance with C")
plt.legend()
plt.show()

#Finding the best value of gamma
classacc = []
m = []
for i in parameters['gamma']:
    svclassifier = SVC(kernel='rbf', C=0.4, random_state=0, gamma = i) 
    svclassifier.fit(X_train2, y_train)
    y_pred = svclassifier.predict(X_test2)
    cm = confusion_matrix(y_test,y_pred)
    classacc.append(accuracy_score(y_test, y_pred))
    accuracies = cross_val_score(estimator = svclassifier, X = X_train2, y = y_train, cv = 10)
    m.append(accuracies.mean())
plt.plot(parameters['gamma'], classacc, label='classification accuracy')
plt.plot(parameters['gamma'], m, label='kfold accuracy')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title("Accuracy variance with gamma")
plt.legend()
plt.show()