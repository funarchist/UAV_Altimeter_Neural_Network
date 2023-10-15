from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import json
import numpy as np
import cv2
import pickle

altitudes = []
image_names = []
altitudes1 = []
image_names1 = []
altitudes2 = []
image_names2 = []
altitudes3 = []
image_names3 = []
altitudes4 = []
image_names4 = []
altitudes5 = []
image_names5 = []
altitudes6 = []
image_names6 = []

arrays = []
list_altitudes = []
list_image_names = []
inputImages = []
trainImagesX1 = []
validationImagesX1= []

with open('../input/auair2019annotations1/annotations.json') as json_file:
    data = json.load(json_file)
    lim1 = data['annotations']
    for p in lim1:
        altitudes.append(p['altitude'])
        image_names.append(p['image_name'])

split = train_test_split(altitudes, image_names, test_size=0.2, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

split = train_test_split(trainAttrX, trainImagesX, test_size=0.2, random_state=42)
(trainAttrX, validationAttrX, trainImagesX, validationImagesX) = split

with open("trainImagesXname.pickle", "wb") as fp:
    pickle.dump(trainImagesX, fp)
with open("validationImagesXname.pickle", "wb") as fp:
    pickle.dump(validationImagesX, fp)
with open("testImagesXname.pickle", "wb") as fp:
    pickle.dump(testImagesX, fp)

for path in trainImagesX:
    if type(path) is str:
        pathim = '../input/auair1/images/' + path
        image = cv2.imread(pathim)
        image = cv2.resize(image, (64, 64))
        trainImagesX1.append(image)

for path in validationImagesX:
    if type(path) is str:
        print("jk")
        pathim = '../input/auair1/images/' + path
        image = cv2.imread(pathim)
        image = cv2.resize(image, (64, 64))
        validationImagesX1.append(image)

with open("trainImagesX.pickle", "wb") as fp:
    pickle.dump(trainImagesX1, fp)

with open("validationImagesX.pickle", "wb") as fp:
    pickle.dump(validationImagesX1, fp)

with open("trainAttrX.pickle", "wb") as fp:
    pickle.dump(trainAttrX, fp)

with open("validationAttrX.pickle", "wb") as fp:
    pickle.dump(validationAttrX, fp)

for alt, img in zip(testAttrX, testImagesX):
    if 2500 < alt < 7500:
        altitudes1.append(alt)
        image_names1.append(img)
    elif 7500 < alt < 12500:
        altitudes2.append(alt)
        image_names2.append(img)
    elif 12500 < alt < 17500:
        altitudes3.append(alt)
        image_names3.append(img)
    elif 17500 < alt < 22500:
        altitudes4.append(alt)
        image_names4.append(img)
    elif 22500 < alt < 27500:
        altitudes5.append(alt)
        image_names5.append(img)
    elif 27500 < alt < 32500:
        altitudes6.append(alt)
        image_names6.append(img)

list_altitudes.append(altitudes1)
list_altitudes.append(altitudes2)
list_altitudes.append(altitudes3)
list_altitudes.append(altitudes4)
list_altitudes.append(altitudes5)
list_altitudes.append(altitudes6)

list_image_names.append(image_names1)
list_image_names.append(image_names2)
list_image_names.append(image_names3)
list_image_names.append(image_names4)
list_image_names.append(image_names5)
list_image_names.append(image_names6)

with open("list_image_names.pickle", "wb") as fp:
    pickle.dump(list_image_names, fp)

with open("list_altitudes.pickle", "wb") as fp:
    pickle.dump(list_altitudes, fp)

with open('list_image_names.pickle', 'rb') as list_image_names:
    list_image_names = pickle.load(list_image_names)

for number_image_names in list_image_names:
    inputImages = []
    for path in number_image_names:
        if type(path) is str:
            print("tjh")
            pathim = '../input/auair1/images/' + path
            image = cv2.imread(pathim)
            image = cv2.resize(image, (64, 64))
            inputImages.append(image)
    ar = np.array(inputImages)
    arrays.append(ar)

with open("images.pickle", "wb") as fp:
    pickle.dump(arrays, fp)