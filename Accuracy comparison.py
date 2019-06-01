#Packages 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Importing dataset
Dataframe_Final = pd.read_csv('Dataset(final).csv')

classacc = {'SVM':0, 'Naive Bayes':0, 'Logistic Regression':0}
m = {'SVM':0, 'Naive Bayes':0, 'Logistic Regression':0}

#Splitting into features and classes
X = Dataframe_Final.drop('File Name',axis=1).drop('Class',axis=1)
y = Dataframe_Final['Class']

#Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 0, shuffle = False)
 
#Scaling features
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train)
X_test2 = sc.transform(X_test)
 
#Training SVM with C as 0.4 and gamma as 0.1
svclassifier = SVC(kernel='rbf', C=0.4, random_state=0, gamma = 0.1) 
svclassifier.fit(X_train2, y_train)
y_pred = svclassifier.predict(X_test2)
cm = confusion_matrix(y_test,y_pred)
accuracies = cross_val_score(estimator = svclassifier, X = X_train2, y = y_train, cv = 10)
m['SVM'] = accuracies.mean()
classacc['SVM'] = accuracy_score(y_test, y_pred)

#Training Gaussian Naive Bayes
nbclassifier = GaussianNB()
nbclassifier.fit(X_train2, y_train)
y_pred = nbclassifier.predict(X_test2)
cm = confusion_matrix(y_test, y_pred)
classacc['Naive Bayes'] = accuracy_score(y_test, y_pred)
accuracies = cross_val_score(estimator = nbclassifier, X = X_train2, y = y_train, cv = 10)
m['Naive Bayes'] = accuracies.mean()

#Training Logistic Regression
lrclassifier = LogisticRegression(random_state = 0)
lrclassifier.fit(X_train2, y_train)
y_pred = lrclassifier.predict(X_test2)
cm = confusion_matrix(y_test,y_pred)
classacc['Logistic Regression'] = accuracy_score(y_test, y_pred)
accuracies = cross_val_score(estimator = lrclassifier, X = X_train2, y = y_train, cv = 10)
m['Logistic Regression'] = accuracies.mean()

#Training Random Forest with 10 trees
rfclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfclassifier.fit(X_train2, y_train)
y_pred = rfclassifier.predict(X_test2)
cm = confusion_matrix(y_test, y_pred)
classacc['rf'] = accuracy_score(y_test, y_pred)
accuracies = cross_val_score(estimator = rfclassifier, X = X_train2, y = y_train, cv = 10)
m['rf'] = accuracies.mean()

#Plotting a graph
raw_data = {'algorithm': list(classacc.keys()),
        'classification accuracy': list(classacc.values()),
        'k-fold accuracy': list(m.values())}
df = pd.DataFrame(raw_data, columns = ['algorithm', 'classification accuracy', 'k-fold accuracy'])
pos = list(range(len(classacc))) 
width = 0.25 
fig, ax = plt.subplots(figsize=(10,5))
plt.bar(pos, df['classification accuracy'], width, alpha=0.5, color='#EE3224', label=df['algorithm'][0]) 
plt.bar([p + width for p in pos], df['k-fold accuracy'], width, alpha=0.5, color='#F78F1E', label=df['algorithm'][1])
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy comparisons')
ax.set_xticks([p + 0.5 * width for p in pos])
ax.set_xticklabels(df['algorithm'])
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(df['classification accuracy'] + df['k-fold accuracy'])] )
plt.legend(['classification accuracy', 'k-fold accuracy'], loc='upper left')
plt.grid()
plt.show()