import numpy as np
import pandas as pd
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,neighbors
from sklearn.preprocessing import StandardScaler

path = "/Users/simon/.cache/kagglehub/datasets/iarunava/cell-images-for-detecting-malaria/versions/1/cell_images"

def load_images_from_folder(folder,label):
    training_data = []
    training_target = []
    for image in os.listdir(folder):
        path = os.path.join(folder,image)
        if path[-3:] == ".db":
            continue
        img = cv.imread(path,cv.IMREAD_GRAYSCALE)
        img = cv.resize(img,(128,128))
        training_data.append(img)
        training_target.append(label)
    return training_data, training_target

infected_data, infected_target = load_images_from_folder(path+"/Parasitized/","infected")
uninfected_data, uninfected_target = load_images_from_folder(path+"/Uninfected/","uninfected")

x = infected_data + uninfected_data
y = infected_target + uninfected_target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

x_train = np.array(x_train,dtype=np.float32) / 255.0
y_train = np.array(y_train)
x_test = np.array(x_test,dtype=np.float32) / 255.0
y_test = np.array(y_test)

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

print(f"accuracy score: {classifier.score(x_test,y_pred)}")
print(metrics.confusion_matrix(y_test,y_predict))