import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data=pd.read_csv("data/Iris.csv")

print(data['species'].value_counts())

data['species']=data['species'].map({'setosa':0,'versicolor':1,
'virginica':2})


x=data.drop('species',axis=1)
y=data['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

ypred=model.predict(x_test)
print(ypred)
print(accuracy_score(y_test,ypred))

joblib.dump(model,'model/random_forest.pkl')
loaded_model=joblib.load('model/random_forest.pkl')
