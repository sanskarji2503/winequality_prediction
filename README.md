# winequality_prediction
Importing Libraries
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')

wine = pd.read_csv("winequality-red.csv")
print("Successfully Imported Data!")
wine.head()

print(wine.shape)

wine.describe(include='all')

print(wine.isna().sum())

wine.corr()
wine.groupby('quality').mean()
sns.countplot(wine['quality'])
plt.show()
sns.countplot(wine['pH'])
plt.show()
sns.countplot(wine['alcohol'])
plt.show()
sns.countplot(wine['alcohol'])
plt.show()
sns.countplot(wine['volatile acidity'])
plt.show()
sns.countplot(wine['citric acid'])
plt.show()

sns.countplot(wine['density'])
plt.show()
sns.kdeplot(wine.query('quality > 2').quality)

sns.distplot(wine['alcohol'])
wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)
wine.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)
wine.hist(figsize=(10,10),bins=50)
plt.show()
wine.hist(figsize=(10,10),bins=50)
plt.show()
sns.pairplot(wine)


sns.violinplot(x='quality', y='alcohol', data=wine)


wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']

wine['goodquality'].value_counts()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))
confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score



Model
Score	
0.893	Random Forest
0.879	Xgboost
0.872	KNN
0.870	Logistic Regression
0.868	SVC
0.864	Decision Tree
0.833	GaussianNB
