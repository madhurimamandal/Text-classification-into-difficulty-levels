#Packages 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
import features as f
 
#Reading the dataset
Dataframe_Final = pd.read_csv('Dataset(final).csv')

#Splitting into features and class
X = Dataframe_Final.drop('File Name',axis=1).drop('Class',axis=1)
y = Dataframe_Final['Class']
 
#Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 0, shuffle = False)

#Feature extraction of input text
f2 = f.features()
col = f2.columns[1:47]
data = []
tlist = []
while True:
    print("Enter the name of a text file or press n to exit")
    fn = input()
    if(fn == 'n'):
        break
    tlist.append(fn)
    fn+='.txt'
    fl = f2.fextr(fn)
    data.append(fl)
for i in range(len(data)):
    data[i] = data[i][1:47]
df = pd.DataFrame(data,columns = col)

#Scaling features
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train)
X_test2 = sc.transform(X_test)
df2 = sc.transform(df)

#Training classifier
svclassifier = SVC(kernel='rbf', C=0.4, random_state=0, gamma = 0.1) 
svclassifier.fit(X_train2, y_train)
y_p = svclassifier.predict(X_test2)

#Predicting class
pred = svclassifier.predict(df2)
col2 = ["Name of text", "Predicted class"]
data2 = []
for i in range(0,len(tlist)):
    x = [tlist[i], pred[i]]
    data2.append(x)
df3 = pd.DataFrame(data2,columns = col2)