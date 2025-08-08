import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,neighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/breast_cancer_wisconson/wbcd_data.csv")

y = df["diagnosis"].to_numpy()

x = df.drop(columns="diagnosis").to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

accuracy = classifier.score(x_test,y_pred)
print(accuracy)