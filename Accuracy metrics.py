#Importing modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Reading the CSV
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

#Training SVM
svclassifier = SVC(kernel='rbf', C=0.4, random_state=0, gamma = 0.1) 
svclassifier.fit(X_train2, y_train)

#Running Kfold
svmaccuracies = cross_val_score(estimator = svclassifier, X = X_train2, y = y_train, cv = 10)
svmaccuracies = [i for i in svmaccuracies]

#Plotting a graph
x = [i for i in range(1,11)]
plt.plot(x,svmaccuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('K-fold accuracy comparisons')
plt.legend()
plt.show()

y_pred = svclassifier.predict(X_test2)
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test, y_pred)
cm = pd.DataFrame(cm)
print(cm)