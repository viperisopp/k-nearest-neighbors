import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,neighbors

df = pd.read_csv("datasets/iris_species.csv")

y = df["Species"].to_numpy()

x = df.drop(columns="Species").to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

accuracy = classifier.score(x_test,y_pred)
print(accuracy)